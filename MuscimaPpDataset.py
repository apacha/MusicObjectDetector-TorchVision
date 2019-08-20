from torch.utils.data import Dataset

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image


class MuscimaPpDataset(Dataset):
    """ The Muscima++ V2.0 Segmentation Dataset

    The [torchvision reference scripts for training object detection, instance segmentation and person keypoint detection](https://github.com/pytorch/vision/tree/v0.3.0/references/detection) allows for easily supporting adding new custom datasets.
    The dataset should inherit from the standard `torch.utils.data.Dataset` class, and implement `__len__` and `__getitem__`.

    The only specificity that we require is that the dataset `__getitem__` should return:

    * image: a PIL Image of size (H, W)
    * target: a dict containing the following fields
        * `boxes` (`FloatTensor[N, 4]`): the coordinates of the `N` bounding boxes in `[x0, y0, x1, y1]` format, ranging from `0` to `W` and `0` to `H`
        * `labels` (`Int64Tensor[N]`): the label for each bounding box
        * `image_id` (`Int64Tensor[1]`): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
        * `area` (`Tensor[N]`): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        * `iscrowd` (`UInt8Tensor[N]`): instances with `iscrowd=True` will be ignored during evaluation.
        * (optionally) `masks` (`UInt8Tensor[N, H, W]`): The segmentation masks for each one of the objects
        * (optionally) `keypoints` (`FloatTensor[N, K, 3]`): For each one of the `N` objects, it contains the `K` keypoints in `[x, y, visibility]` format, defining the object. `visibility=0` means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt `references/detection/transforms.py` for your new keypoint representation

    If your model returns the above methods, they will make it work for both training and evaluation, and will use the evaluation scripts from pycocotools.

    Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratio), then it is recommended to also implement a `get_height_and_width` method, which returns the height and the width of the image. If this method is not provided, we query all elements of the dataset via `__getitem__` , which loads the image in memory and is slower than if a custom method is provided.

    """

    def __init__(self, image_directory: str, masks_directory: str, transforms=None, cache_images_in_memory=False):
        self.masks_directory = masks_directory
        self.image_directory = image_directory
        self.transforms = transforms
        self.shrink_factor = 3
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(image_directory)))
        self.masks = list(sorted(os.listdir(masks_directory)))
        self.cache_images_in_memory = cache_images_in_memory
        if cache_images_in_memory:
            self.image_cache = []
            for i in range(len(self.imgs)):
                self.image_cache.append(self.load_image(i))
            self.target_cache = []
            for i in range(len(self.masks)):
                self.target_cache.append(self.load_target(i))


    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if self.cache_images_in_memory:
            img = self.image_cache[idx]
        else:
            img = self.load_image(idx)

        if self.cache_images_in_memory:
            target = self.target_cache[idx]
        else:
            target = self.load_target(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def load_target(self, idx):
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask_path = os.path.join(self.masks_directory, self.masks[idx])
        mask = Image.open(mask_path)  # type:Image.Image
        mask = mask.resize((mask.width // self.shrink_factor, mask.height // self.shrink_factor))

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def load_image(self, idx):
            img_path = os.path.join(self.image_directory, self.imgs[idx])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img.width // self.shrink_factor, img.height // self.shrink_factor))
            return img

    def __len__(self) -> int:
        return len(self.imgs)
