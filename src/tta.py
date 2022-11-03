import torch
import torchvision.transforms.functional as F

__all__ = ["TestTimeAug"]


class TestTimeAug():
    def __init__(self, models, transforms, **kwargs):
        self.models = models
        self.transforms = transforms
    
    def get_transfoms_data(self, data):
        transform_funs = {
            'identity': lambda x: x,
            'hflip': F.hflip,
            'vflip': F.vflip,
            'hvflip': lambda x: F.vflip(F.hflip(x)),
            'vhflip': lambda x: F.hflip(F.vflip(x))
        }
        data_list = [transform_funs[f](data) for f in self.transforms]

        return data_list

    def get_avg(self, tensor):
        return torch.mean(tensor, dim=0)
    
    def get_class(self, prob):
        return prob.argmax(dim=1)

    def predict(self, data):
        data_list = self.get_transfoms_data(data)
        pred_list = [m(d) for m in self.models for d in data_list]
        pred_prob = self.get_avg(torch.stack(pred_list, dim=0))
        pred = self.get_class(pred_prob)

        return pred
