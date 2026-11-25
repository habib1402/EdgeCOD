"""COD datasets. Only images that have a ground-truth mask are used."""
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def list_pairs(img_dir, gt_dir):
    """(image, mask) path pairs, matched by file stem and sorted by stem."""
    images = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)}
    gts = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.lower().endswith('.png')}
    stems = sorted(images.keys() & gts.keys())
    return [(os.path.join(img_dir, images[s]), os.path.join(gt_dir, gts[s])) for s in stems]


def split_holdout(pairs, frac, seed):
    """Deterministic split of the training set into (train, VAL), used for every design decision (Section 4.1).

    frac=0.1, seed=0 on the 4,040 CAMO+COD10K training images gives the 404-image VAL split listed in splits/val.txt.
    """
    if frac <= 0:
        return pairs, []
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(frac * len(pairs)))
    val, train = sorted(idx[:n_val]), sorted(idx[n_val:])
    return [pairs[i] for i in train], [pairs[i] for i in val]


def to_tensor(img):
    return torch.from_numpy(((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


class TrainSet(Dataset):
    """Random crop (side fraction in `crop`) + horizontal flip, resized to `size`.

    The training loop then rescales each batch to a size drawn from data.scales (multi-scale training).
    """

    def __init__(self, pairs, size=448, crop=(0.75, 1.0), hflip=0.5):
        self.pairs, self.size, self.crop, self.hflip = pairs, size, crop, hflip

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, gt_path = self.pairs[i]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        if gt.size != img.size:
            gt = gt.resize(img.size, Image.NEAREST)
        w, h = img.size
        s = random.uniform(*self.crop)
        cw, ch = max(1, int(w * s)), max(1, int(h * s))
        x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        box = (x0, y0, x0 + cw, y0 + ch)
        img, gt = img.crop(box), gt.crop(box)
        if random.random() < self.hflip:
            img, gt = img.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        gt = gt.resize((self.size, self.size), Image.BILINEAR)
        mask = torch.from_numpy(np.asarray(gt, dtype=np.float32) / 255.0)[None]
        return to_tensor(img), mask


class TestSet(Dataset):
    """Images resized to `size` x `size`; masks kept at their original size for scoring.

    `pairs` may also hold (image, None) for inference without ground truth.
    """

    def __init__(self, pairs, size=416):
        self.pairs, self.size = pairs, size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, gt_path = self.pairs[i]
        img = Image.open(img_path).convert('RGB')
        hw = (img.size[1], img.size[0])
        img = img.resize((self.size, self.size), Image.BILINEAR)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if gt_path else None
        name = os.path.splitext(os.path.basename(gt_path or img_path))[0]
        return to_tensor(img), gt, name, hw


def test_collate(batch):
    img, gt, name, hw = batch[0]
    return img[None], gt, name, hw


def test_pairs(root, name):
    d = os.path.join(root, 'TestDataset', name)
    return list_pairs(os.path.join(d, 'Imgs'), os.path.join(d, 'GT'))
