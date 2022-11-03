import os
import torch

from tqdm import tqdm

from .logger import Logger
from .metric import compute_acc, compute_wp_f1
from .utils import Recoder, save_topk_ckpt
from .constant import WEIGHT_DIR, LOG_PATH, BASELINE_F1_SCORE

__all__ = ["Trainer"]


class Trainer():
    def __init__(self,
                 model,
                 device,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 args):

        self.args = args
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log = Logger(LOG_PATH)
        self.weight_dir = WEIGHT_DIR

    def train_step(self):
        self.model.train()
        recorder = Recoder(self.cur_ep, 'train')
        train_bar = tqdm(self.train_loader, desc=f'Training {self.cur_ep}')

        for i, data in enumerate(train_bar):
            pred, label, loss, acc = self.__share_step(data)

            loss.backward()
            if (i+1) % self.args.accumulate_grad_bs == 0:
                self.__update_grad()

            recorder.update(
                loss=loss.item(),
                acc=acc.item(),
                bs=pred.shape[0],
                lr=self.lr_scheduler.get_last_lr()[0])

            train_bar.set_postfix(recorder.get_iter_record())
            del pred, label

        self.log.add(**recorder.get_epoch_record())
        train_bar.close()

        return

    @torch.no_grad()
    def val_step(self):
        self.model.eval()
        recorder = Recoder(self.cur_ep, 'val')
        val_bar = tqdm(self.val_loader, desc=f'Validation {self.cur_ep}')
        pred_list, label_list = [], []

        for data in val_bar:
            pred, label, loss, acc = self.__share_step(data)
            recorder.update(
                loss=loss.item(),
                acc=acc.item(),
                bs=pred.shape[0],
                lr=self.lr_scheduler.get_last_lr()[0])

            pred_list.extend(pred.argmax(dim=1).cpu().tolist())
            label_list.extend(label.cpu().tolist())
            val_bar.set_postfix(recorder.get_iter_record())

        f1_dict, wp = compute_wp_f1(pred_list, label_list)   
        self.log.add(**recorder.get_epoch_record())

        if min(f1_dict.values()) >= BASELINE_F1_SCORE:
            save_topk_ckpt(self.model, self.cur_ep, wp, self.weight_dir, topk=5)

        val_bar.close()

        return

    def __share_step(self, data):
        image, label = data['image'], data['label']
        image = image.to(self.device)
        label = label.to(self.device)
        pred = self.model(image)
        loss = self.criterion(pred, label)
        acc = compute_acc(pred, label)

        return pred, label, loss, acc

    def set_model_device(self):
        self.model = self.model.data_parallel(self.args.device)
        self.model.to(self.device)

        return

    def __update_grad(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        return

    def fit(self):
        self.set_model_device()
        save_topk_ckpt(self.model, 0, 0, self.weight_dir, topk=5)

        for self.cur_ep in range(1, self.args.epoch+1):
            self.train_step()
            self.val_step()
            self.log.save()

        return
