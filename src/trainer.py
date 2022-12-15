from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler

from .logger import Logger, GradientNorm, TensorboardWriter
from .metric import compute_acc, Evaluator
from .utils import Recorder, save_topk_ckpt
from .constant import WEIGHT_DIR, BASELINE_F1_SCORE, SAVE_TOPK


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
                 *arg, **kwarg
                 ):

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_num = len(train_loader)
        self.val_num = len(val_loader)
        self.criterion = criterion
        self.optimizer = optimizer
        self.amp = amp
        self.grad_scaler = GradScaler(amp)
        self.accum_grad_bs = accum_grad_bs
        self.clip_grad = clip_grad
        self.lr_scheduler = lr_scheduler
        self.sample_weight = None
        self.evaluator = Evaluator()
        self.recorder = Recorder()
        self.logger = Logger()
        self.grad_norm_fn = GradientNorm()
        self.train_tensorboard = TensorboardWriter('train')
        self.val_tensorboard   = TensorboardWriter('valid')
        self.weight_dir = WEIGHT_DIR

    def train_step(self):
        self.model.train()
        self.evaluator.reset()
        self.recorder.reset('train', self.cur_ep)
        train_bar = tqdm(self.train_loader, desc=f'Training {self.cur_ep}')

        for step, data in enumerate(train_bar, start=1):
            loss = self.share_step(data)
            self.update_model(loss, step)
            self.train_tensorboard.add(
                step + (self.cur_ep - 1) * self.train_num,
                **self.recorder.get_record()
                )
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
            self.share_step(data)
            self.val_tensorboard.add(
                step + (self.cur_ep - 1) * self.val_num,
                **self.recorder.get_record()
                )
            val_bar.set_postfix(self.recorder.get_iter_record())
        
        self.evaluator.compute_wp()
        self.logger.add(**self.recorder.get_epoch_record(), wp=self.evaluator.wp)

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

    def set_model_device(self):
        self.model.to(self.device)
        return

    def share_step(self, data):
        image, label = data['image'], data['label']
        image = image.to(self.device)
        label = label.to(self.device)
        pred = self.model(image)
        loss = self.criterion(pred, label)
        acc = compute_acc(pred, label)

        self.evaluator.add(pred.cpu(), label.cpu())
        self.recorder.update(
            loss=loss.item(),
            acc=acc.item(),
            bs=pred.shape[0],
            lr=self.lr_scheduler.get_last_lr()[0],
            )

        return loss

    def update_model(self, loss, step):
        if self.amp:
            self.grad_scaler.scale(loss / self.accum_grad_bs).backward()
            grad_norm = self.grad_norm_fn(self.model)
            self.recorder.grad_norm.update(grad_norm, 1)

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
            grad_norm = self.grad_norm_fn(self.model)
            self.recorder.grad_norm.update(grad_norm, 1)

            if step % self.accum_grad_bs == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

        return 


    def fit(self, epoch):
        self.set_model_device()
        save_topk_ckpt(self.model, 0, 0, self.weight_dir, topk=SAVE_TOPK)

        for self.cur_ep in range(1, epoch+1):
            self.train_step()
            self.val_step()
            self.logger.save()

        return
