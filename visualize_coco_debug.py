#!/usr/bin/env python3
"""
visualize_coco_debug.py

Sovrappone annotazioni COCO (bbox e segmentation) su N immagini dalla cartella Data
e salva le immagini annotate in una cartella scelta dall'utente.

Uso esempio:
python visualize_coco_debug.py --coco annotations.json --data_dir Data --outdir annotated_out --num 20

Dipendenze opzionali: opencv-python, pillow, pycocotools (per RLE decode)
pip install opencv-python pillow pycocotools
"""

import os
import json
import argparse
import random
from collections import defaultdict

try:
    import cv2
except Exception as e:
    raise SystemExit("opencv-python is required (pip install opencv-python)")

import numpy as np

# Try to import pycocotools for RLE masks
mask_utils = None
try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def random_color(seed_val=0):
    random.seed(seed_val)
    return tuple(int(random.random() * 255) for _ in range(3))


def draw_annotations(img, anns, categories_map, seed_base=0):
    overlay = img.copy()
    h, w = img.shape[:2]
    for i, ann in enumerate(anns):
        color = random_color(seed_base + ann.get('id', i))
        # Draw bbox
        bbox = ann.get('bbox')
        if bbox and len(bbox) == 4:
            x, y, bw, bh = map(int, bbox)
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, thickness=2)
            # label background
            label = categories_map.get(ann.get('category_id'), str(ann.get('category_id')))
            txt = f"{label} {ann.get('id','') }"
            # compute text size
            (txw, txh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(overlay, (x, y - txh - 6), (x + txw + 4, y), color, -1)
            cv2.putText(overlay, txt, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Draw segmentation (polygons or RLE)
        seg = ann.get('segmentation')
        if seg:
            if isinstance(seg, list):
                # list of polygons
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], isClosed=True, color=(0,0,0), thickness=1)
            elif isinstance(seg, dict):
                # RLE
                if mask_utils is not None:
                    try:
                        m = mask_utils.decode(seg)
                        if m.ndim == 3:
                            m = m[:, :, 0]
                        mask_indices = m.astype(bool)
                        # color overlay
                        overlay[mask_indices] = (overlay[mask_indices] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                    except Exception:
                        # fallback: skip RLE if decode fails
                        pass
                else:
                    # cannot decode RLE without pycocotools
                    pass

    # alpha blend
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def build_image_ann_map(coco):
    imgs = {img['id']: img for img in coco.get('images', [])}
    cats = {cat['id']: cat['name'] for cat in coco.get('categories', [])}
    ann_map = defaultdict(list)
    for ann in coco.get('annotations', []):
        ann_map[ann['image_id']].append(ann)
    return imgs, ann_map, cats


def main():
    parser = argparse.ArgumentParser(description='Visual debug COCO annotations')
    parser.add_argument('--coco', default='annotations.json', help='COCO annotations JSON (default: annotations.json in project root)')
    parser.add_argument('--data_dir', default='Data', help='Directory with images (default: Data)')
    parser.add_argument('--outdir', required=True, help='Where to save annotated images')
    parser.add_argument('--num', type=int, default=10, help='Number of images to render')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle images before selecting')
    args = parser.parse_args()

    if not os.path.exists(args.coco):
        raise SystemExit(f"COCO file not found: {args.coco}")

    with open(args.coco, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    imgs, ann_map, cats = build_image_ann_map(coco)
    image_list = list(imgs.values())
    if not image_list:
        raise SystemExit('No images found in COCO file')

    random.seed(args.seed)
    if args.shuffle:
        random.shuffle(image_list)

    selected = image_list[:args.num]
    os.makedirs(args.outdir, exist_ok=True)

    for img_info in selected:
        fname = img_info.get('file_name')
        if not fname:
            print(f"Skipping image without file_name: {img_info}")
            continue
        # try path in data_dir
        img_path = os.path.join(args.data_dir, fname)
        if not os.path.exists(img_path):
            # try as absolute or relative to project root
            if os.path.exists(fname):
                img_path = fname
            else:
                print(f"Image not found, skipping: {img_path}")
                continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        anns = ann_map.get(img_info['id'], [])
        annotated = draw_annotations(img, anns, cats, seed_base=args.seed)
        # save with same filename into outdir
        out_path = os.path.join(args.outdir, os.path.basename(fname))
        # use imencode + tofile to handle unicode paths on Windows
        ext = os.path.splitext(out_path)[1]
        if ext == '':
            out_path = out_path + '.png'
        is_success, encimg = cv2.imencode(ext if ext.startswith('.') else '.'+ext, annotated)
        if is_success:
            encimg.tofile(out_path)
            print(f"Saved: {out_path}")
        else:
            print(f"Failed to encode/save: {out_path}")


if __name__ == '__main__':
    main()
