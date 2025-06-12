
## 🔧 Code Structure Highlights

* 🧮 **Loss Function**
  Implemented in [`utils.py`](./utils.py) for supervising keypoint prediction accuracy.

* 🔥 **Keypoint → Heatmap Conversion**
  A **Gaussian-based method** is used to convert keypoints into pseudo-heatmaps.
  Implemented in the `KeypointToHeatMap` class within [`transforms.py`](./transforms.py).

* 📂 **Dataset: `DatasetSeg.py`**
  Defines the `LeafKeypoint` class for loading wheat flag leaf images and keypoints.

  * 📄 **Annotation Loader**: Loads data from `labels.xlsx`
  * 🔀 **Data Split**: Automatically partitions into training / validation / test sets
  * 🎨 **Augmentation**: Applies random HSV color jitter during training
  * 🔧 **Transforms Support**: Compatible with external transforms; includes custom `collate_fn` for DataLoader

  📌 **Keypoints Format**
  `[[leaf_tip_x, leaf_tip_y], [joint_x, joint_y], [stem_x, stem_y]]` (3×2 array)



