"""Predict camouflage maps for your own images (no ground truth needed).

  python predict.py --config configs/edgecod_s.yaml --weights weights/edgecod_s_s2026.pth \
      --input path/to/images --out path/to/maps [--overlay]

Writes one 8-bit PNG probability map per image, at the image's original size. --overlay also writes the image
with the predicted object tinted red (threshold 0.5).
"""
import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from edgecod.config import load_config
from edgecod.data import IMG_EXTS, TestSet, test_collate
from edgecod.evaluate import predict
from edgecod.model import EdgeCOD, load_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--input', required=True, help='an image file or a folder of images')
    p.add_argument('--out', required=True)
    p.add_argument('--overlay', action='store_true')
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_weights(EdgeCOD.from_config(cfg['model'], pretrained=False), args.weights).to(device).eval()

    if os.path.isdir(args.input):
        files = sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(IMG_EXTS))
    else:
        files = [args.input]
    os.makedirs(args.out, exist_ok=True)
    loader = DataLoader(TestSet([(f, None) for f in files], cfg['data']['test_size']), batch_size=1,
                        num_workers=2, collate_fn=test_collate)
    for (img, _, name, hw), path in zip(loader, files):
        pred = predict(model, img, hw, device)
        cv2.imwrite(os.path.join(args.out, name + '.png'), pred)
        if args.overlay:
            im = cv2.imread(path)
            tint = im.copy()
            tint[pred >= 128] = (0.5 * tint[pred >= 128] + np.array([0, 0, 127])).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, name + '_overlay.jpg'), tint)
    print(f'{len(files)} maps written to {args.out}')


if __name__ == '__main__':
    main()
