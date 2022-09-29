# Crop_Classification
2022/09/29 新增train_num valid_num 可以決定要放多少訓練資料驗證資料進去訓練(基於同一個fold_0不變下)
---

HackMD https://hackmd.io/@sam23582211/SkNkYCaWj
```
   ├── dataset
   │   │   ├── longan
   │   │   │   ├── 00a3ef6f-0493-4698-b453-3498c93d10c6.jpg
   │   │   ...
   │   │ 
   │   │   ├── passionfruit
   │   │   │   ├── 00d38376-19a1-4fa0-831b-561c42d5efe6.jpg
   │   │   ...
   │   │   
   │   ├── ...
   │   │   ├── roseapple
   │   │   │   ├── 00a2c035-c5c4-4b2f-aca9-c3f4f99de193.jpg
   │   │    ...
   ├── src
   │   ├── transform.py
   │   ├── dataset.py
   │   ├── ...
   ├── index
   │   ├── fold_0.csv (Train : Valid = 8:1)
   │   ├── fold_1.csv (Train : Valid : Test = 0.05:0.01:0.94) for toy case test
   ├── Step0_split_data.py
   ├── train.py
   ├── train.sh
   │   ...

```
Step0_split_data : 執行此程式前確保有將解壓縮資料放入dataset中~


### 這醜醜圖片之後會拿掉
![image](https://user-images.githubusercontent.com/93210989/192424422-f5863734-4a9d-4023-9fa8-add7c4d1741d.png)  
