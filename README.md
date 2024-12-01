# PARF-Net

![framework](E:\texlive\PARF-Net\figure1.png)

we develop a new
method PARF-Net to integrate convolutions of Pixel-wise
Adaptive Receptive Fields (Conv-PARF) into hybrid Network
for medical image segmentation.The features derived from the Conv-PARF lay-
ers are further processed using hybrid Transformer-CNN
blocks under a lightweight manner, to effectively capture
local and long-range dependencies, thus boosting the seg-
mentation performance.

## Requirements

```angular2html
Python 3.8+.
```

## Installation

### Clone repository

```angular2html
git clone https://github.com/zhyu-lab/parfnet
cd parfnet
```

### Create conda environment (optional)

```angular2html
conda create --name parfnet python=3.8
```

Then activate it:

```angular2html
conda activate parfnet
```

## Install requirements

```angular2html
numpy 1.24.4
torch 2.4.1
tqdm 4.67.1
compressai 1.2.6
timm 1.0.11
torchmetrics 1.5.2
opencv-python 4.10.0.84
scikit-learn 1.3.2
tensorboard 2.14.0
```

## Usage

*Note: If you have some problems with the code, the [issues](https://github.com/McGregorWwww/UCTransNet/issues?q=is%3Aissue+is%3Aclosed) may help.*

### 1. Data Preparation

#### 1.1. GlaS  MoNuSeg  DSB2018 Datasets

The original data can be downloaded in following links:

* MoNuSeg Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)
* GLAS Dataset - [Link (Original)](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest)
* DSB2018 Dataset -[Link (Original](https://www.kaggle.com/c/data-science-bowl-2018/data)

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── GlaS
    │   ├── images
    │   │   └──test_*.bmp  train_*.bmp
    │   │   
    │   ├── masks
    │   │   └── test_*.bmp  train_*.bmp       
    ├── MoNuSeg
    │   ├── images
    │   │   ├── train_images
    │   │   └── test_images
    │   ├── masks
    │   │   ├── train_masks
    │   │   └── test_images
    └── DSB2018
        ├── images
        │   ├── *.png
        └── masks
            └── *.png
       


       
```

#### 1.2. Synapse Dataset

The Synapse dataset we used is provided by TransUNet's authors.
Please go to [https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md)
for details.

### 2. Training

Since we didn't use any pre-trained models, we just run mian.py directly

python main.py

#### 

### 3. Testing

#### 3.1. Get Pre-trained Models

Since training on the Synapse dataset is time-consuming, if you don't want to train the model yourself, you can download it at the following link: https://drive.google.com/file/d/1s97JYmOWyY4S7NvfIl1fbwm2FUdQwkvo/view?usp=drive_link Once downloaded, you can use the TransUNet code to test our model.

#### 3.2. Test the Model and Visualize the Segmentation Results

For GlaS,MoNuSeg,Dsb2018 we test it during training, and also get metrics such as Iou Dice, and visualization of the segmentation results

You can get the Dice and IoU scores and the visualization results. 

### ## Reference

* Attention U-Net: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
* MultiResUNet: https://github.com/makifozkanoglu/MultiResUNet-PyTorch
* TransUNet: https://github.com/Beckschen/TransUNet
* Swin-Unet: https://github.com/HuCaoFighting/Swin-Unet
* MedT: https://github.com/jeya-maria-jose/Medical-Transformer
* UcTransUet:[UCTransNet/README.md at main · McGregorWwww/UCTransNet](https://github.com/McGregorWwww/UCTransNet)

## Contact

Xu  Ma ([lebronmx@stu.nxu.edu.cn](lebronmx@stu.nxu.edu.cn))
