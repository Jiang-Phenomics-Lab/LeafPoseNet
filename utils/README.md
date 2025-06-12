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

**Keypoints format:**
A NumPy array of shape (3×2):
`[[leaf_tip_x, leaf_tip_y], [joint_x, joint_y], [stem_x, stem_y]]`

