## Code Structure Highlights

### Keypoint to Heatmap Conversion

A Gaussian-based method is used to convert keypoints into pseudo-heatmaps.
This function is implemented in the `KeypointToHeatMap` class in [`transforms.py`](./transforms.py).

### Loss Function

The loss function used to supervise keypoint prediction accuracy is implemented in [`utils.py`](./utils.py).

### Dataset (`DatasetSeg.py`)

Defines the `LeafKeypoint` dataset class for wheat flag leaf angle keypoint detection.

* Loads annotations from `labels.xlsx`
* Automatically splits the dataset into training, validation, and test sets
* Applies random HSV augmentation during training to enhance robustness
* Supports custom transform pipelines and provides a `collate_fn` for PyTorch DataLoader compatibility

### Annotation Format (`labels.xlsx`)

The annotation file is a tabular `.xlsx` file with the following columns:

| img\_name               | num  | id    | img\_path                              | t\_angle | t\_x0 | t\_x1 | t\_x2 | t\_y0 | t\_y1 | t\_y2 |
| ----------------------- | ---- | ----- | -------------------------------------- | -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| C001\_1\_20230508081541 | C001 | 1588B | /media/.../C001\_1\_20230508081541.jpg | 146.6695 | 1529  | 1723  | 1729  | 1818  | 1538  | 1290  |

* `t_x0`, `t_y0`: Coordinates of the **leaf**
* `t_x1`, `t_y1`: Coordinates of the **joint**
* `t_x2`, `t_y2`: Coordinates of the **stem**
* `t_angle`: Flag leaf angle (in degrees)



