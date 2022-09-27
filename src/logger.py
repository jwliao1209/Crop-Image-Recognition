import csv


class Logger():
    def __init__(self, path, print_=False):
        self.path = path
        self.records = []
        self.print_ = print_

    def add(self, **inputs):
        self.records.append(inputs)
        if self.print_:
            self.print(**inputs)

        return

    def print(self, **inputs):
        print(', '.join(f"{k}: {v}" for k, v in zip(inputs.keys(), inputs.values())))

        return

    def save(self):
        fieldnames = self.records[0].keys()
        with open(self.path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        return
