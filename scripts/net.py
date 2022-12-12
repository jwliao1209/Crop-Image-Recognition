import os
from typing import Any, Sequence
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.data import DataLoader, Dataset, CacheDataset
from sklearn.metrics import accuracy_score
from metrics.f1 import get_wp_f1
from torchvision.models import (densenet121,
                                DenseNet121_Weights,
                                swin_v2_t,
                                Swin_V2_T_Weights,
                                swin_s,
                                Swin_S_Weights,
                                swin_v2_s,
                                Swin_V2_S_Weights,
                                swin_b,
                                Swin_B_Weights,
                                regnet_y_16gf,
                                RegNet_Y_16GF_Weights,
                                efficientnet_b0,
                                EfficientNet_B0_Weights,
                                MaxVit)
from pytorch_lightning import LightningModule, LightningDataModule
from transforms import train_transforms, val_transforms
from loss.angular_loss import AngularPenaltySMLoss
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

def get_model(name):
    supported_model = {
        "swin_v2_t": swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1),
        "swin_s": swin_s(weights=Swin_S_Weights.IMAGENET1K_V1),
        "swin_v2_s": swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1),
        "densenet": densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
        "regnet": regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1),
        "swin_b":swin_b(weights=Swin_B_Weights.IMAGENET1K_V1),
        "efficient": efficientnet_b0(EfficientNet_B0_Weights.IMAGENET1K_V1),
        "maxvit_512": MaxVit(input_size=(512, 512), 
                    stem_channels=64,
                    partition_size=16,
                    block_channels=[64, 128, 256, 512],
                    block_layers=[2, 2, 5, 2],
                    head_dim=32,
                    stochastic_depth_prob=0.2),
    }
    return supported_model[name.lower()]

def get_loss(name):
    supported_loss = {
        "BCE": CrossEntropyLoss(),
        "FL" : torch.hub.load('adeelh/pytorch-multi-class-focal-loss', 
                                model='FocalLoss', 
                                gamma=2, 
                                reduction='mean'),
    }
    return supported_loss[name.upper()]

def get_optimizer(name):
    supported_optimizer = {
        "adam": Adam,
        "adamw": AdamW,
    }
    return supported_optimizer[name.lower()]

def get_scheduler(optimizer, scheduler_config):
    supported_scheduler = {
        "cosine": CosineAnnealingLR(optimizer, 
                                    T_max = scheduler_config.get("T_max", 50),
                                    eta_min = scheduler_config.get("eta_min", 0)),
        "warmcosine": LinearWarmupCosineAnnealingLR(optimizer,
                                                    max_epochs=scheduler_config.get("max_epochs"),
                                                    warmup_epochs=scheduler_config.get("warmup_epochs"))
    }
    return supported_scheduler[scheduler_config["name"].lower()]

def get_dataset(data, transform, cache=True):
    if cache is True:
        return CacheDataset(data=data, transform = transform, cache_rate=1, num_workers=8)
    else:
        return Dataset(data=data, transform = transform)


# ----------------------------------------------------------------

class LitModel(LightningModule):
    def __init__(self, config):
        super(LitModel, self).__init__()
        self.save_hyperparameters()
        self.model = get_model(config["model"]["backbone"].lower())
        
        self.linear = nn.Linear(config["model"]["backbone_num_features"], config["model"]["num_classes"])
        self.loss_function = get_loss(config["loss"])
        self.use_additional_loss = config.get("use_additional_loss", False)
        self.additional_loss = AngularPenaltySMLoss(in_features=1000, out_features=33, loss_type="arcface")

    def forward(self, x : torch.Tensor) :
        feats = self.model(x)
        y = self.linear(feats)
        return feats, y

    def configure_optimizers(self):
        optimizer =  get_optimizer(self.hparams.config["optimizer"])(
            self.parameters(),
            lr = self.hparams.config["lr"],
        )

        #test sam
        # optimizer = get_optimizer(self.hparams.config["optimizer"])(
        #     self.parameters(),
        #     AdamW,
        #     lr=self.hparams.config["lr"]
        # )

        scheduler = get_scheduler(optimizer, self.hparams.config["scheduler"])

        return [optimizer], [scheduler]
    
    def share_step(self, batch : Any, batch_idx : int) :
        images, labels = batch["image"], batch["label"]
        feats, output = self.forward(images)
        labels = labels.long()

        main_loss = self.loss_function(output, labels)
        if self.use_additional_loss:
            additional_loss = self.additional_loss(feats, labels)
            loss = (main_loss + additional_loss)/2
        else:
            additional_loss = 0
            loss = main_loss

        preds = torch.argmax(output, dim=1)
        
        result = {"loss" : loss, 
                  "main_loss": main_loss, 
                  "additional_loss":additional_loss,
                  "output" : preds.detach(), 
                  "labels" : labels.detach()}
        return result


    def epoch_end(self, outputs, prefix):
        epsilon = 1e-8
        loss_result  = torch.mean(torch.stack([o["loss"] for o in outputs])).detach().cpu()
        main_loss_result  = torch.mean(torch.stack([o["main_loss"] for o in outputs])).detach().cpu()
        if self.use_additional_loss:
            additional_loss_result  = torch.mean(torch.stack([o["additional_loss"] for o in outputs])).detach().cpu()
        else:
            additional_loss_result = 0

        y       = torch.cat([o["labels"] for o in outputs]).cpu()
        y_pred  = torch.cat([o["output"].detach() for o in outputs]).cpu()

        acc = accuracy_score(y, y_pred)
        f1_score, _ = get_wp_f1(y, y_pred)
        mean_f1 = np.mean(list(f1_score.values()))

        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/loss", loss_result, prog_bar=True)
        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/main_loss", main_loss_result, prog_bar=True)
        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/addtional_loss", additional_loss_result, prog_bar=True)
        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/acc", acc, prog_bar=True)
        self.log('step', self.trainer.current_epoch)
        self.log(f"{prefix}/mean_f1", mean_f1, prog_bar=True)


    def training_step(self, batch : Any, batch_idx : int) -> Any :
        return self.share_step(batch, batch_idx)

    def training_epoch_end(self, outputs : Sequence) -> Any :
        self.epoch_end(outputs,"training")

    def validation_step(self, batch : Any, batch_idx : int) -> Any:
        return self.share_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: Sequence) -> Any :
        self.epoch_end(outputs, "val")

    def test_step(self, batch : Any, batch_idx : int) -> Any:
        return self.share_step(batch, batch_idx, prefix="test")

    def test_epoch_end(self, outputs: Sequence) -> Any :
        self.epoch_end(outputs, "test")
        
    def predict_step(self, batch, batch_idx):
        feats, y = self(batch["image"])
        output = torch.argmax(y, dim=1)
        return {"output": output, "prob":y}

# ----------------------------------------------------------------

def concat_path(datalist: list, dataroot: str):
    subsets = [s for s in ["training", "validation", "test"] if s in datalist]
    for subset in subsets:
        for i in datalist[subset]:
            i["image"] = os.path.join(dataroot, i["image"])

    return datalist

class DataModule(LightningDataModule):
    def __init__(self, **data_config):
        super(DataModule, self).__init__()
        self.batch_size = data_config["batch_size"]
        self.val_batch_size = data_config.get("val_batch_size", 1)
        self.dataroot = data_config["dataroot"]
        self.datalist = data_config["datalist"]
        self.transforms_config = data_config["transforms_config"]
        self.cache = data_config["cache"]

    def setup(self, stage=None): 
        data_list = json.load(open(self.datalist))
        data_list = concat_path(data_list, self.dataroot)

        train_files  = data_list["training"]
        val_files = data_list["validation"]

        self.train_ds = get_dataset(data=train_files, transform = train_transforms(**self.transforms_config), cache=self.cache)
        self.val_ds = get_dataset(data=val_files, transform = val_transforms(**self.transforms_config), cache=self.cache)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size = self.batch_size,
                          num_workers = 16,
                          shuffle=True,
                          )

    def val_dataloader(self): 
        return DataLoader(self.val_ds,
                          batch_size = self.val_batch_size,
                          num_workers = 16,
                          )
    
