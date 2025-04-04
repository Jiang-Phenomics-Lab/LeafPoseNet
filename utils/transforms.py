import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    output_flipped = torch.flip(output_flipped, dims=[3])

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped

def adjust_image(image, fixed_size):
    h, w, c = image.shape
    xmax = w
    ymax = h
    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:

        wi = h / hw_ratio
        pad_w = int((wi - w) / 2)
        xmax = int(w + 2*pad_w)
        new_image = np.zeros((h, xmax, c))
        new_image[:, pad_w:w+pad_w, :] = image
    elif h / w < hw_ratio:

        hi = w * hw_ratio
        pad_h = int((hi - h) / 2)
        ymax = int(h + 2*pad_h)
        new_image = np.zeros((ymax, w, c))
        new_image[pad_h:h+pad_h, :, :] = image
    else:
        new_image = image

    return new_image

def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column
    preds[:, :, 1] = torch.floor(idx / w)  # row 

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    assert trans is not None
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone().cpu().numpy()
    
    preds = preds*2


    return preds, maxvals.detach().cpu().numpy()


def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],      
                    [w - 1, 0],  
                    [0, h - 1]], 
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:

        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]          
        dst[1, :] = [size[1] - pad_w - 1, 0] 
        dst[2, :] = [pad_w - 1, size[0] - 1] 
    else:

        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]           
        dst[1, :] = [size[1] - 1, pad_h - 1] 
        dst[2, :] = [0, size[0] - pad_h - 1] 

    trans = cv2.getAffineTransform(src, dst) 

    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(resize_img)
    # plt.show()

    dst /= 2  
    reverse_trans = cv2.getAffineTransform(dst, src) 

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):

    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:

        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:

        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):

    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Blur(object):
    def __call__(self, image, target):
        kernel_size = np.random.randint(1, 5+1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image, target

class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []


            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])


            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps


            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                if w > 1 and h > 1:
 
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""
    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (256, 192)):  #e.g. (768, 576)
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        image = adjust_image(img, self.fixed_size)
        h, w, c = image.shape
        scale_y = self.fixed_size[0] / h

        image = cv2.resize(image, (self.fixed_size[1], self.fixed_size[0]), interpolation=cv2.INTER_NEAREST)


        dst_center = np.array([(self.fixed_size[0] - 1) / 2, (self.fixed_size[1] - 1) / 2])

        scale = 1.0
        angle = 0

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  
        
        M = cv2.getRotationMatrix2D((dst_center[1], dst_center[0]), angle, scale)
        resize_img = cv2.warpAffine(image, M, (self.fixed_size[1], self.fixed_size[0]))



        if "keypoints" in target:
            kps = target["keypoints"].copy()
            kps = kps*scale_y
            kps = (kps+ 0.5).astype(np.int_) 
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], M)
            target["keypoints_new"] = kps

        if np.any((target['keypoints_new'][:, 0] < 2) | (target['keypoints_new'][:, 0] > self.fixed_size[1]-2) | (target['keypoints_new'][:, 1] < 2) | (target['keypoints_new'][:, 1] > self.fixed_size[0]-2)):
            resize_img = image
            kps = target["keypoints"].copy()
            kps = kps*scale_y
            kps = (kps+ 0.5).astype(np.int_) 
            target["keypoints_new"] = kps

        target["trans"] = M
        target["reverse_trans"] = 0
        return resize_img, target
    

class RandomHorizontalFlip(object):

    def __init__(self, p: float = 0.5, matched_parts: list = None):
        # assert matched_parts is not None
        self.p = p
        # self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = target["keypoints_new"]
            # visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            target["keypoints_new"] = keypoints
            # target["visible"] = visible

        return image, target


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (768 // 4, 576 // 4),
                 gaussian_sigma: int = 2
                 ):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma #2
        self.kernel_radius = self.sigma * 3# self.sigma * 3
        #self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = [1, 1, 1]

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1 #13
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2

        kernel1 = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel1[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))#(2 * 1.5 ** 2))
        kernel2 = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2   #x_center=6

        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel2[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))#(2 * 1.5 ** 2))
        kernel3 = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel3[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))#(2 * 1.5 ** 2))


        self.kernels = [kernel1, kernel2, kernel3]


    def generate_distance_weights(self, hw, center_point, ellipse_axes=(15, 40), angle_degrees=45):
        x, y = np.meshgrid(np.arange(hw[0]), np.arange(hw[1]))
        

        x_rot = (x - center_point[0]) * np.cos(np.radians(angle_degrees)) + (y - center_point[1]) * np.sin(np.radians(angle_degrees))
        y_rot = -(x - center_point[0]) * np.sin(np.radians(angle_degrees)) + (y - center_point[1]) * np.cos(np.radians(angle_degrees))
        
  
        distances = np.sqrt((x_rot / ellipse_axes[0])**2 + (y_rot / ellipse_axes[1])**2)
        
  
        scaled_weights = np.piecewise(distances,
                                    [distances <= 9, (distances > 9) & (distances <= 30), distances > 30],
                                    [lambda d: 0*d+1, lambda d: 0*d+2, lambda d: 0*d+3])

        return scaled_weights

    def __call__(self, image, target):
        kps = target["keypoints_new"]
        num_kps = kps.shape[0]
        kps_weights = self.kps_weights
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        weights = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 2 + 0.5).astype(np.int_) 

        
        for kp_id,kernel in zip(range(num_kps),self.kernels):
        #for kp_id in num_kps:    
            kernel_radius = self.kernel_radius
            #kernel = self.kernel

            if kp_id==1:
                slope = (heatmap_kps[2][1] - heatmap_kps[1][1])/(heatmap_kps[2][0] - heatmap_kps[1][0]+np.spacing(1))
                angle_radians = np.arctan(slope)
            else:
                slope = (heatmap_kps[kp_id][1] - heatmap_kps[1][1])/(heatmap_kps[kp_id][0] - heatmap_kps[1][0]+np.spacing(1))
                angle_radians = np.arctan(slope)
                



            x, y = heatmap_kps[kp_id]
            ul = [x - kernel_radius, y - kernel_radius]  # up-left x,y
            br = [x + kernel_radius, y + kernel_radius]  # bottom-right x,y

            # Usable gaussian range
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]
            #heatmap[kp_id]/=heatmap[kp_id].sum() 
            weights[kp_id] = self.generate_distance_weights((self.heatmap_hw[1], self.heatmap_hw[0]), heatmap_kps[kp_id], (1, 3.3), np.degrees(angle_radians)+90)
        



        # plot_heatmap(image, heatmap, kps, kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["weights"] = torch.as_tensor(weights, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target