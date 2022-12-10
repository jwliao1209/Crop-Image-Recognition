# Crop_Classification

<!-- ### 2022/09/29 新增 `--train_num` `--valid_num` 可以決定要放多少訓練資料驗證資料進去訓練(基於同一個fold_0不變下) -->

HackMD : https://hackmd.io/@sam23582211/SkNkYCaWj  
CSV record : https://docs.google.com/spreadsheets/d/184lMG54i9AlbHJtmzxLVR5S3WXctsajP2TbE0wgKWwc/edit#gid=0  

```
   ├── dataset
   │   │   ├── mask
   │   │   │   ├── train_mask
   │   │   │   ├── public_mask
   │   │   │   └── private_mask
   │   │   ├── train
   │   │   │   ├── asparagus
   │   │   │   ├── bambooshoots
   │   │   │   ├── ...
   │   │   │   └── waterbamboo
   │   │   ├── public
   │   │   │   ├── 0
   │   │   │   ├── 1
   │   │   │   ├── ...
   │   │   │   ├── a
   │   │   │   ├── b
   │   │   │   ├── ...
   │   │   │   └── f
   │   │   └── private
   ├── src
   │   ├── transform.py
   │   ├── dataset.py
   │   └── ...
   ├── index
   │   ├── fold_0.csv (Train : Valid = 8:1)
   │   ├── fold_1.csv (Train : Valid : Test = 0.05:0.01:0.94) for toy case test
   ├── Step0_split_data.py
   ├── train.py
   ├── train.sh
   └──   ...

```
Step0_split_data : 執行此程式前確保有將解壓縮資料放入dataset中~


### 這醜醜圖片之後會拿掉
![image](https://user-images.githubusercontent.com/93210989/192424422-f5863734-4a9d-4023-9fa8-add7c4d1741d.png)  
