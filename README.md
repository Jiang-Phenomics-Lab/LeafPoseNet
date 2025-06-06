# LeafPoseNet a low-cost, high-accuracy method for estimating flag leaf angle in wheat
## Introduction
- We defined flag leaf angle measurement as a keypoint-based pose detection task and proposed a lightweight deep learning model, LeafPoseNet, to accurately detect these keypoints. 
- The flag leaf angle in a 2D image can be calculated based on the positions of three keypoints: the flag leaf center (Point L), the junction between the flag leaf and stem (Point J), and the stem center (Point S). These keypoints define the geometric and topological relationships necessary for angle computation. 
- To reduce the effects of minor annotation errors on model training when using discrete keypoints as targets, we transformed keypoint prediction into a heatmap regression task. 


## Installation

### Clone the Repository: 
```
git clone https://github.com/Jiang-Phenomics-Lab/LeafPoseNet.git
cd LeafPoseNet
```

## Requirements
Install dependencies using pip
```
pip install -r requirements.txt
```

## Train:
`train.py` is used to train segmentation models
```bash
python train.py
```
## Predict
`predict.py` is used to segment wheat plants from images
```bash
python predict.py
```
## Phenotyping:
The phenotyping method is in the `phenotypying.ipynb` 

## Datasets
Images collected in 2023 and 2024 are available for download at the following link:
[https://1024terabox.com/s/1XzcIaxilrahdC-3xoJOLVg](https://1024terabox.com/s/1Z0rS-m-MDbEI9kl_nR2_Lw)

## Model training strategies and runtime environment
The LANetLeafPoseNet model was implemented in Python 3.8 using the Pytorch framework and trained on an NVIDIA 4080 GPU. The AdamW optimizer was used, with a learning rate warm-up by a cosine annealing strategy. The initial learning rate was set to 2 × 10-3, with a minimum learning rate of 1 × 10-6, and a batch size of 4. Training was conducted for a total of 500 epochs, and the model achieving the best accuracy on the validation set was selected as the final model. 


