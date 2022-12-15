import os
import torch

from torch.utils.tensorboard import SummaryWriter
from .utils import save_csv
from .constant import LOG_PATH, TENSORBORD_DIR


class Logger():
    def __init__(self, path=LOG_PATH, print_=False):
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
    def __init__(self, mode, save_dir=TENSORBORD_DIR):
        path = os.path.join(save_dir, mode)
        self.writer = SummaryWriter(path)
        os.makedirs(path, exist_ok=True)

    def add(self, step, **inputs):
        for k, v in inputs.items():
            self.writer.add_scalar(k, v, step)
        return


class GradientNorm():
    def __init__(self, norm_type=2.0):
        super().__init__()
        self.norm_type = norm_type

    def __call__(self, model):
        # Extract gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads:
            device = grads[0].device
            # Calculate norm (same as torch.nn.utils.clip_grad_norm_)
            norm_type = float(self.norm_type)
            total_norm = torch.norm(
                torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
                norm_type
            )
            return total_norm.item()
        
        else:
            return 0
