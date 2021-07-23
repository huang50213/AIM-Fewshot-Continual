# Few-Shot Learning with Attentive Independent Mechanisms

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

<img src="attention-AIM.png" width="600">

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

All experiments can be run on a single NVIDIA GTX1080Ti GPU.


The code was tested with python3.6 the following software versions:

| Software        | version | 
| ------------- |-------------| 
| cuDNN         | v7500 |
| Pytorch      | 1.5.0  |
| CUDA | v10.0    |


## Training

### Pretrain features

Model: `ConvNet_4_64` and `WRN_28_10`

Dataset: `miniImageNet` and `Cifar`
```
python3 train_feat.py --outDir pretrained_model/miniImageNet_WRN_60Epoch_base --cuda --dataset miniImageNet  --nbEpoch 60 --config config/WRN_miniImageNet_1shot.yaml --useAIM
```


### Meta-learning

```
 python3 main.py --config config/Conv_miniImageNet_1shot.yaml --seed 100 --gpu 3 --useAIM
```



## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

> 📋Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

> 📋Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> 📋Pick a licence and describe how to contribute to your code repository. 

