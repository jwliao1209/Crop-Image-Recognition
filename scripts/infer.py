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


def compute_metrics(args):
    config = yaml.full_load(open(args.config))
    datalist = json.load(open(config["data_config"]["datalist"]))
    data_list = concat_path(datalist, config["data_config"]["dataroot"])

    val_label = [i["label"] for i in datalist["validation"]]
    test_label = [i["label"] for i in datalist["test"]]
    val_pred = pd.read_csv("../infer/val_prob_"+config["name"]+".csv")["pred"].tolist()
    test_pred = pd.read_csv("../infer/test_prob_"+config["name"]+".csv")["pred"].tolist()
    
    val_acc = accuracy_score(val_label, val_pred)
    test_acc = accuracy_score(test_label, test_pred)
    val_f1, _ = get_wp_f1(val_label, val_pred)
    test_f1, _ = get_wp_f1(test_label, test_pred)
    val_mean_f1 = np.mean(list(val_f1.values()))
    test_mean_f1 = np.mean(list(test_f1.values()))


    print(f"val acc : {val_acc}")
    print(f"val mean f1: {val_mean_f1}")
    print(f"test acc : {test_acc}")
    print(f"test mean f1: {test_mean_f1}")

    config["acc"] = {"val_acc": float(val_acc), "test_acc":float(test_acc)}
    config["f1"] = {"val_f1": float(val_mean_f1), "test_f1":float(test_mean_f1)}
    yaml.dump(config, open(args.config, "w"), sort_keys=False)

def predict(trainer, model, loader, savename, prefix, data_id):
    outputs  = trainer.predict(model, loader)
    y_pred  = torch.cat([o["output"] for o in outputs])
    y_pred = y_pred.cpu().numpy()
    results = [[data_id[i], int(y_pred[i])] for i in range(len(y_pred))]
    name = ['id', 'pred']
    infer_result = pd.DataFrame(columns=name, data = results)
    infer_result.to_csv('../infer/'+prefix+'_prob_'+savename+'.csv',index=None)
    
def main(args):
    config = yaml.full_load(open(args.config))
    data_list = json.load(open(config["data_config"]["datalist"]))
    data_list = concat_path(data_list, config["data_config"]["dataroot"])

    val_files = data_list["validation"]
    test_files = data_list["test"]

    val_id = [i["image"].split('/')[-1].split('.')[0] for i in val_files]
    test_id = [i["image"].split('/')[-1].split('.')[0] for i in test_files]

    val_ds = Dataset(data=val_files, transform=val_transforms(**config["data_config"]["transforms_config"]))
    test_ds = Dataset(data=test_files, transform=val_transforms(**config["data_config"]["transforms_config"]))

    val_loader = DataLoader(val_ds, batch_size=config["data_config"]["batch_size"], num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=config["data_config"]["batch_size"], num_workers=8)
    
    trainer  = Trainer(logger=False, 
                        gpus=1, 
                        accelerator="ddp", 
                        num_sanity_val_steps=0)
    
    model = LitModel.load_from_checkpoint(config["ckpt"], strict=False)

    predict(trainer, model, val_loader, config["name"], "val", val_id)
    print("End validation prediction ...")
    predict(trainer, model, test_loader, config["name"], "test", test_id)
    print("End testing prediction ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config_path")
    args=parser.parse_args()
    main(args)
    compute_metrics(args)