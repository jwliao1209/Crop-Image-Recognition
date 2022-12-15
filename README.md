# Crop_Classification
## How to train the model?
* `net.py` 中已經預設幾個 model 可供選擇
* Example: 在 command 資料夾中執行 `source train.sh ../config/regnet` 即可訓練
  * 需要注意的事：datalist 與 data 需要先準備好，並修改 config file 裡的路徑


### YAML 檔結構
```python
name: swin_s_1080 （跟 log, checkpoint 資料夾名字有關）
savepath: v4 （跟 log, checkpoint 資料夾名字有關）
epoches: 150
strategy: ddp
accumulate_batches: 1
precision: 16
gradient_clip_val: 5.0
data_config:
  batch_size: 16
  val_batch_size: 64
  dataroot: /neodata/pathology_breast/aidea/dataset/ （預期上會接到你 datalist 中每個影像的前面）
  datalist: /neodata/pathology_breast/aidea/crop_code/datalist/fold_0.json
  cache: false （如果 ram > 500GB 才考慮使用）
  transforms_config:
    img_size: 1080
    random_transform: true （auto augmentation）
    random_rotation: false
model_config:
  model:
    num_classes: 33
    backbone: swin_s
    backbone_num_features: 1000
  loss: FL (Focal loss)
  use_additional_loss: false (Augular Loss 此次沒使用到)
  optimizer: adamw
  lr: 0.0001
  scheduler:
    name: cosine
    T_max: 50
    eta_min: 0
ckpt: ../checkpoint/swin_s_1080.ckpt
```
### Datalist 結構
最終會是一份 json 檔，一個 `dict`，有三個 key: "training", "validation", "test"，每個 key 所對應的 value 是一個 `list` ，`list` 裡面有多個 `dict`，每個 `dict` 包含影像位置及 label
```python
{
    "training": [
        {
            "image": "asparagus/000b43a3-d331-47ad-99a4-4c0fa9b48298.jpg",
            "label": 0
        },
        {
            "image": "asparagus/00172189-3156-48d7-bb1e-f0a922bc54b8.jpg",
            "label": 0
        },
        ...
```

## Inference 方法

* 準備好訓練時使用到的 config file (`.yaml`), 在 `infer_public.py` 給 config file 位置即可（需要特別注意 infer_public 內有一行需要針對使用情況做修正， `data_list = json.load(open("../datalist/public_private.json"))` 請針對要 inference 的 data 給予正確的 json 檔）
    * json 檔裡面是一個 `list` 結構，每一個元素皆是一個 `dict` ，形式為 {"image": image_path}, image_path 應替換成檔案的絕對路徑
* 請在 config file 內指定你要的 checkpoint 路徑
* 使用範例: 進入到 command 資料夾中，執行 `source infer_public.sh ../config/regnet.yaml` (可以在 `.sh` 裡面調整使用哪張 GPU)

## Contact
有關此份 code 的操作問題可以聯絡: rex19981002@gmail.com