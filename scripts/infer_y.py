import os
import json
import yaml
import argparse

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from metrics.f1 import get_wp_f1
from pytorch_lightning import Trainer
from net import LitModel, concat_path
from transforms import val_transforms


def predict(trainer, model, loader, savename, prefix, data_id):
    outputs  = trainer.predict(model, loader)
    y_pred  = torch.cat([o["prob"] for o in outputs])
    y_pred = y_pred.cpu().numpy()
    results = [[data_id[i], y_pred[i]] for i in range(len(y_pred))]
    name = ['id', 'prob']
    infer_result = pd.DataFrame(columns=name, data = results)
    infer_result.to_csv('../infer/'+prefix+'_tsne_'+savename+'.csv',index=None)
    
def main(args):
    config = yaml.full_load(open(args.config))
    data_list = json.load(open(config["data_config"]["datalist"]))
    data_list = concat_path(data_list, config["data_config"]["dataroot"])

    val_files = data_list["validation"]
    test_files = data_list["test"]

    val_id = [i["image"].split('/')[-1].split('.')[0] for i in val_files]

    val_ds = Dataset(data=val_files, transform=val_transforms(**config["data_config"]["transforms_config"]))

    val_loader = DataLoader(val_ds, batch_size=config["data_config"]["batch_size"], num_workers=8)
    
    trainer  = Trainer(logger=False, 
                        gpus=1, 
                        accelerator="ddp", 
                        num_sanity_val_steps=0)
    
    model = LitModel.load_from_checkpoint(config["ckpt"])

    predict(trainer, model, val_loader, config["name"], "val", val_id)
    print("End validation prediction ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config_path")
    args=parser.parse_args()
    main(args)