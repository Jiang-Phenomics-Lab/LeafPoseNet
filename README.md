# LANet a low-cost, high-accuracy method for estimating flag leaf angle in wheat
## Introduction
- LANet is a lightweight model that reliable measurements of leaf angle across diverse wheat.
- To reduce the effects of minor annotation errors on model training when using discrete keypoints as targets, we transformed keypoint prediction into a heatmap regression task. 



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

Green leaves and yellow leaves segmentation
![LeavesSeg](./assets/seg.gif)






