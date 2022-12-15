# Crop_Classification


## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/Crop_Classification.git
```

## Download checkpoint
https://drive.google.com/drive/folders/1cZTBzgOuuf3ms6V__71nWlOXtvmT0TxZ?usp=sharing


## Training
To train the model, you can run this command:
```
python train.py -ep <epoch> \
                -bs <batch size> \
                --model <model> \
                --image_size <crop and resize to img_size> \
                --loss <loss function> \
                --optim <optimizer> \
                --lr <learning rate> \
                --weight_decay <parameter weight decay> \
                --scheduler <learning rate schedule> \
                --auto_aug <use auto-augmentation> \
                --num_workers <number worker> \
                --device <gpu id>
```


## Inference
To get the inference results, you can run this command:
```
python inference.py --checkpoint <MONTH-DAY-HOUR-MIN-SEC> --topk <number of model you want to ensemble>
```


## Experiment results
<table>
  <tr>
    <td>model</td>
    <td>size</td>
    <td>bs</td>
    <td>loss</td>
    <td>optimizer</td>
    <td>scheduler</td>
    <td>public WP</td>
  </tr>
  <tr>
    <td>EfficientNet-B0</td>
    <td>1080</td>
    <td>20</td>
    <td>FL</td>
    <td>AdamW</td>
    <td>Step decay</td>
    <td></td>
  </tr>
<table>


## Citation
```
@misc{
    title  = {crop_classification},
    author = {Jia-Wei Liao, Yen-Jia Chen, Yi-Chen Hung, Jing-En Hung, Shang-Yen Lee},
    url    = {https://github.com/Jia-Wei-Liao/Crop_Classification},
    year   = {2022}
}
```
