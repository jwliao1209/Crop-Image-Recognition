# Crop_Classification


## Getting the code
You can download all the files in this repository by cloning this repository:  
```
git clone https://github.com/Jia-Wei-Liao/Crop_Classification.git
```

## Setting the environment
To set the environment, you can run this command:
```
pip install -r requirements.txt
```

## Download the checkpoint
You can download the dataset and checkpoint from our Google Drive:
https://drive.google.com/drive/folders/1cZTBzgOuuf3ms6V__71nWlOXtvmT0TxZ?usp=sharing


## Training
To train the model, you can run this command:
```
bash script/train.sh
```
or
```
python train.py -ep <epoch> \
                -bs <batch size> \
                --model <model> \
                --fold <k fold validation data> \
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
bash script/infer.sh
```
or
```
python inference.py --checkpoint <MONTH-DAY-HOUR-MIN-SEC> \
                    --topk <number of model you want to ensemble>
```

<!--
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
-->


## Reproducing submission
To reproduce our submission, please do the following steps:
1. [Getting the code](https://github.com/Jia-Wei-Liao/Crop_Classification/#Getting-the-code)
2. [Setting the environment](https://github.com/Jia-Wei-Liao/Crop_Classification/#Setting-the-environment)
3. [Download dataset and checkpoint](https://github.com/Jia-Wei-Liao/Crop_Classification/#Download-the-checkpoint)
4. [Inference](https://github.com/Jia-Wei-Liao/Crop_Classification/#Inference)


## Operating System and Device
We develop the code on Ubuntu 22.04 operating system and use python 3.9.15 version. All trainings are performed on a server with a single NVIDIA 3090 GPU.


## Citation
```
@misc{
    title  = {crop_classification},
    author = {Jia-Wei Liao, Yen-Jia Chen, Yi-Cheng Hung, Jing-En Hung, Shang-Yen Lee},
    url    = {https://github.com/Jia-Wei-Liao/Crop_Classification},
    year   = {2022}
}
```
