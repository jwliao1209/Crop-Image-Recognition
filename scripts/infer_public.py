import json
import yaml
import argparse

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from net import LitModel
from transforms import val_transforms

mapping = {0: 'asparagus',
 1: 'bambooshoots',
 2: 'betel',
 3: 'broccoli',
 4: 'cauliflower',
 5: 'chinesecabbage',
 6: 'chinesechives',
 7: 'custardapple',
 8: 'grape',
 9: 'greenhouse',
 10: 'greenonion',
 11: 'kale',
 12: 'lemon',
 13: 'lettuce',
 14: 'litchi',
 15: 'longan',
 16: 'loofah',
 17: 'mango',
 18: 'onion',
 19: 'others',
 20: 'papaya',
 21: 'passionfruit',
 22: 'pear',
 23: 'pennisetum',
 24: 'redbeans',
 25: 'roseapple',
 26: 'sesbania',
 27: 'soybeans',
 28: 'sunhemp',
 29: 'sweetpotato',
 30: 'taro',
 31: 'tea',
 32: 'waterbamboo'}

def predict(trainer, model, loader, savename, data_id):
    outputs  = trainer.predict(model, loader)
    y_pred  = torch.cat([o["output"] for o in outputs])
    y_pred = y_pred.cpu().numpy()
    results = [[data_id[i], int(y_pred[i])] for i in range(len(y_pred))]
    name = ['filename', 'label']
    infer_result = pd.DataFrame(columns=name, data = results)
    infer_result["label"] = infer_result["label"].map(mapping)
    infer_result = infer_result.sort_values(by="filename")
    infer_result.to_csv('../private/'+savename+'.csv',index=None)


def main(args):
    config = yaml.full_load(open(args.config))
    data_list = json.load(open("../datalist/public_private.json"))
    ids = [i["image"].split("/")[-1] for i in data_list]

    ds = Dataset(data=data_list, transform=val_transforms(**config["data_config"]["transforms_config"]))
    loader = DataLoader(ds, batch_size=config["data_config"]["batch_size"], num_workers=8)

    trainer = Trainer(logger=False, 
                        gpus=1, 
                        accelerator="ddp",
                        num_sanity_val_steps=0)

    model = LitModel.load_from_checkpoint(config["ckpt"], strict=False)

    predict(trainer, model, loader, config["name"], ids)
    print("End prediction ...")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    args=parser.parse_args()
    main(args)
