import os

from torch.utils.tensorboard import SummaryWriter
from .utils import save_csv
from .constant import TENSORBORD_DIR


class Logger():
    def __init__(self, path, print_=False):
        self.path = path
        self.records = []
        self.print_ = print_

    def add(self, **inputs):
        self.records.append(inputs)
        self.print_(**inputs) if self.print_ else None
        return

    def save(self):
        save_csv(self.path, self.records)
        return


class TensorboardWriter():
    def __init__(self):
        self.writer = SummaryWriter(TENSORBORD_DIR)
        os.makedirs(TENSORBORD_DIR, exist_ok=True)
    
    def add(self, step, **inputs):
        for k, v in inputs.items():
            self.writer.add_scalar(k, v, step)
        return
