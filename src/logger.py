from src.utils import save_csv

__all__ = ['Logger']


class Logger():
    def __init__(self, path, print_=False):
        self.path = path
        self.records = []
        self.print_ = print_

    def add(self, **inputs):
        self.records.append(inputs)
        self.print_(**inputs) if self.print_ else None
        return

    def print_(self, **inputs):
        print(', '.join(f"{k}: {v}" for k, v in zip(inputs.items())))
        return

    def save(self):
        save_csv(self.path, self.records)
        return
