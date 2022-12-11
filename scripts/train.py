import os
from typing import Any, Sequence
import json
import yaml
import argparse
import torch

from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from net import LitModel, DataModule


seed_everything(42, workers=True)
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    config = yaml.full_load(open(args.config))
    print("CONFIG".center(100, "*"))
    print(yaml.dump(config, sort_keys=False))
    print("*"*100)
    
    lit_model = LitModel.load_from_checkpoint(
        args.ckpt,
        strict=False
        ) if args.ckpt else LitModel(config["model_config"])

    datamodule = DataModule(**config["data_config"])
    save_filename = config["savepath"]
    version_name = config["name"]
    
    checkpoint = ModelCheckpoint(
        dirpath = '../checkpoint/'+save_filename+'/'+version_name,
        filename = 'acc',
        monitor = 'val/acc',
        save_top_k=1,
        mode = 'max',
        save_last = True,
        verbose = True)

    checkpoint_f1 = ModelCheckpoint(
        dirpath = '../checkpoint/'+save_filename+'/'+version_name,
        filename = 'f1',
        monitor = 'val/mean_f1',
        save_top_k=1,
        mode = 'max',
        save_last = True,
        verbose = True)
    
    checkpoint_loss   = ModelCheckpoint(
        dirpath='../checkpoint/'+save_filename+'/'+version_name,
        filename="best_loss",
        monitor="val/loss",
        mode="min",
        verbose=True,
        save_last=True)
    
    lr_monitor = LearningRateMonitor()

    log_dir = '../log/'
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, 
                                             name=save_filename, 
                                             version=version_name)
    trainer = Trainer(
        max_epochs=config["epoches"],
        gpus=-1,
        strategy=config.get("strategy", "dp"),
        callbacks = [checkpoint, checkpoint_f1, checkpoint_loss, lr_monitor],
        logger = tb_logger,
        num_sanity_val_steps=0 ,
        enable_checkpointing=True,
        deterministic=True,
        accumulate_grad_batches = config.get("accumulate_batches", None),
        precision=config.get("precision", 32),
        gradient_clip_val=config.get("gradient_clip_val", 0))

    trainer.fit(lit_model, 
                datamodule=datamodule, 
                ckpt_path=args.ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config_path")
    parser.add_argument("--ckpt", type=str)
    args=parser.parse_args()
    main(args)
