from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler

from .logger import Logger, TensorboardWriter
from .metric import compute_acc, Evaluator
from .utils import Recorder, save_topk_ckpt
from .constant import WEIGHT_DIR, LOG_PATH, BASELINE_F1_SCORE, SAVE_TOPK


class Trainer():
    def __init__(self,
                 model,
                 device,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 accum_grad_bs,
                 lr_scheduler,
                 amp=False,
                 clip_grad=None,
                 ):

        self.model = model
        self.device = device
        self.amp = amp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_num = len(train_loader)
        self.val_num = len(val_loader)
        self.criterion = criterion
        self.optimizer = optimizer
        self.grad_scaler = GradScaler(amp)
        self.accum_grad_bs = accum_grad_bs
        self.clip_grad = clip_grad
        self.lr_scheduler = lr_scheduler
        self.sample_weight = None
        self.evaluator = Evaluator()
        self.recorder = Recorder()
        self.logger = Logger(LOG_PATH)
        self.tensorboard = TensorboardWriter()
        self.weight_dir = WEIGHT_DIR

    def train_step(self):
        self.model.train()
        self.evaluator.reset()
        self.recorder.reset('train', self.cur_ep)
        train_bar = tqdm(self.train_loader, desc=f'Training {self.cur_ep}')

        for step, data in enumerate(train_bar, start=1):
            loss = self.__share_step(data, step, self.train_num)
            self.__update_model(loss, step)
            train_bar.set_postfix(self.recorder.get_iter_record())

        self.evaluator.compute_wp()
        self.logger.add(**self.recorder.get_epoch_record(), wp=self.evaluator.wp)
        train_bar.close()

        return

    @torch.no_grad()
    def val_step(self):
        self.model.eval()
        self.evaluator.reset()
        self.recorder.reset('val', self.cur_ep)
        val_bar = tqdm(self.val_loader, desc=f'Validation {self.cur_ep}')

        for step, data in enumerate(val_bar):
            self.__share_step(data, step, self.val_num)
            val_bar.set_postfix(self.recorder.get_iter_record())
        
        self.evaluator.compute_wp()
        self.logger.add(**self.recorder.get_epoch_record(), wp=self.evaluator.wp)
        
        # self._update_sample_weight()
        print(self.evaluator.f1_dict, self.evaluator.wp)

        if self.evaluator.get_min_f1() >= BASELINE_F1_SCORE:
            save_topk_ckpt(
                self.model,
                self.cur_ep,
                self.evaluator.wp,
                self.weight_dir,
                topk=SAVE_TOPK
                )

        val_bar.close()
        return

    def __set_model_device(self):
        self.model.to(self.device)
        return

    def __share_step(self, data, step, data_num):
        image, label = data['image'], data['label']
        image = image.to(self.device)
        label = label.to(self.device)
        pred = self.model(image)
        loss = self.criterion(pred, label)# , self.sample_weight)
        acc = compute_acc(pred, label)

        record = dict(
            loss=loss.item(),
            acc=acc.item(),
            bs=pred.shape[0],
            lr=self.lr_scheduler.get_last_lr()[0]
            )

        self.evaluator.add(pred.cpu(), label.cpu())
        self.recorder.update(record)
        self.tensorboard.add(
            step + (self.cur_ep - 1) * data_num,
            **record
        )
        return loss

    def __update_model(self, loss, step):
        if self.amp:
            self.grad_scaler.scale(loss / self.accum_grad_bs).backward()
            if step % self.accum_grad_bs == 0:

                if self.clip_grad is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(),
                        clip_value=self.clip_grad
                        )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        else:
            (loss / self.accum_grad_bs).backward()
            if step % self.accum_grad_bs == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step() # self.cur_ep + step / len(self.train_loader)

        return

    # TODO: update sample weight by valid map
    def _update_sample_weight(self):
        f1_scores = torch.tensor(list(self.evaluator.f1_dict.values()))
        weight = 1 / f1_scores
        self.sample_weight = weight / torch.sum(weight)
        self.sample_weight = self.sample_weight.to(self.device)
        print(self.sample_weight)

        return

    def fit(self, epoch):
        self.__set_model_device()
        save_topk_ckpt(self.model, 0, 0, self.weight_dir, topk=SAVE_TOPK)

        for self.cur_ep in range(1, epoch+1):
            self.train_step()
            self.val_step()
            self.logger.save()

        return
