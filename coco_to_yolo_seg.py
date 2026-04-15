#!/usr/bin/env python3
"""
coco_to_yolo_seg.py

Converte annotazioni COCO (poligoni o RLE) in formato YOLO (segmentazione) compatibile con Ultralytics (es. YOLOv2.6/YOLOv8-seg).
Crea la struttura:
out_dir/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml

Esempio:
python coco_to_yolo_seg.py --coco annotations.json --src_dir Data\\frame_tesi --out_dir dataset_yolo --train_ratio 0.8 --seed 42

Dipendenze opzionali: pycocotools (per decodificare RLE), opencv-python, pillow
pip install opencv-python pillow pycocotools
"""

import os
import json
import argparse
import random
import shutil
from collections import defaultdict

try:
    import cv2
except Exception:
    raise SystemExit('opencv-python è richiesto: pip install opencv-python')

import numpy as np

# Try to import pycocotools for RLE masks
mask_utils = None
try:
    from pycocotools import mask as mask_utils
except Exception:
    mask_utils = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def coco_category_mapping(categories):
    # Map COCO category ids to consecutive 0..N-1
    coco_ids = [c['id'] for c in categories]
    coco_ids_sorted = sorted(coco_ids)
    mapping = {cid: i for i, cid in enumerate(coco_ids_sorted)}
    names = [None] * len(coco_ids_sorted)
    for c in categories:
        names[mapping[c['id']]] = c.get('name', str(c['id']))
    return mapping, names


def seg_to_polygons_from_rle(rle, height, width):
    # Return list of polygons (each polygon as list of x,y floats)
    polys = []
    if mask_utils is None:
        return polys
    try:
        m = mask_utils.decode(rle)
        if m is None:
            return polys
        if m.ndim == 3:
            m = m[:, :, 0]
        mask = (m.astype('uint8') * 255).astype('uint8')
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            cnt = cnt.reshape(-1, 2)
            # flatten
            poly = cnt.flatten().tolist()
            polys.append(poly)
    except Exception:
        return polys
    return polys


def normalize_poly(poly, width, height):
    # poly: list of floats [x1,y1,x2,y2,...]
    norm = []
    for i, v in enumerate(poly):
        if i % 2 == 0:
            norm.append(v / float(width))
        else:
            norm.append(v / float(height))
    return norm


def main():
    parser = argparse.ArgumentParser(description='Convert COCO -> YOLO segmentation format')
    parser.add_argument('--coco', default='annotations.json', help='COCO JSON file')
    parser.add_argument('--src_dir', default=os.path.join('Data', 'frame_tesi'), help='Cartella immagini sorgente')
    parser.add_argument('--out_dir', default='dataset_yolo', help='Cartella di output (images/labels/data.yaml)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio train set (default 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--copy_images', action='store_true', help='Copia le immagini nella cartella di destinazione (default: False, scambia con symlink se preferisci)')
    args = parser.parse_args()

    if not os.path.exists(args.coco):
        raise SystemExit(f'COCO file non trovato: {args.coco}')
    if not os.path.exists(args.src_dir):
        raise SystemExit(f'Directory immagini sorgente non trovata: {args.src_dir}')

    with open(args.coco, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = {im['id']: im for im in coco.get('images', [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get('annotations', []):
        anns_by_image[ann['image_id']].append(ann)

    cat_map, names = coco_category_mapping(coco.get('categories', []))

    # Prepare output folders
    images_train = os.path.join(args.out_dir, 'images', 'train')
    images_val = os.path.join(args.out_dir, 'images', 'val')
    labels_train = os.path.join(args.out_dir, 'labels', 'train')
    labels_val = os.path.join(args.out_dir, 'labels', 'val')
    for p in [images_train, images_val, labels_train, labels_val]:
        ensure_dir(p)

    img_list = list(images.values())
    random.seed(args.seed)
    random.shuffle(img_list)

    n_train = int(len(img_list) * args.train_ratio)
    train_imgs = set([i['id'] for i in img_list[:n_train]])

    def write_label_file(label_path, ann_lines):
        if not ann_lines:
            # create empty label file to signal no objects (Ultralytics tolerates empty files)
            open(label_path, 'w', encoding='utf-8').close()
            return
        with open(label_path, 'w', encoding='utf-8') as lf:
            for line in ann_lines:
                lf.write(line + '\n')

    for im in img_list:
        fid = im['id']
        fname = im.get('file_name')
        width = im.get('width')
        height = im.get('height')
        if not fname or width is None or height is None:
            print(f"Skipping image missing metadata: {im}")
            continue
        src_path = os.path.join(args.src_dir, fname)
        if not os.path.exists(src_path):
            # try filename as-is
            if os.path.exists(fname):
                src_path = fname
            else:
                print(f"Immagine non trovata, salto: {src_path}")
                continue

        split = 'train' if fid in train_imgs else 'val'
        dest_img_dir = images_train if split == 'train' else images_val
        dest_label_dir = labels_train if split == 'train' else labels_val

        # copy image
        dest_img_path = os.path.join(dest_img_dir, os.path.basename(fname))
        if args.copy_images:
            try:
                shutil.copy2(src_path, dest_img_path)
            except Exception as e:
                print(f"Errore copiando immagine {src_path} -> {dest_img_path}: {e}")
                continue
        else:
            # create small text file with path pointing to original (ultralytics can accept absolute paths in data.yaml, but here we copy path references)
            # We'll still copy images to keep a self-contained dataset by default in future runs; for now create a symlink if possible
            try:
                if os.name == 'nt':
                    # On Windows creating symlink requires admin; fallback to copy
                    shutil.copy2(src_path, dest_img_path)
                else:
                    os.symlink(os.path.abspath(src_path), dest_img_path)
            except Exception:
                try:
                    shutil.copy2(src_path, dest_img_path)
                except Exception as e:
                    print(f"Impossibile linkare o copiare immagine {src_path}: {e}")
                    continue

        # build label lines
        ann_lines = []
        anns = anns_by_image.get(fid, [])
        for ann in anns:
            cid = ann.get('category_id')
            if cid not in cat_map:
                # skip unknown category
                continue
            class_idx = cat_map[cid]
            seg = ann.get('segmentation')
            if not seg:
                # no segmentation -> skip (YOLO-seg needs polygon)
                continue
            if isinstance(seg, list):
                # one or more polygons
                for poly in seg:
                    if len(poly) < 6:
                        continue
                    norm = normalize_poly(poly, width, height)
                    # join as space separated floats with max 6 decimals
                    pts_str = ' '.join([f"{x:.6f}" for x in norm])
                    ann_lines.append(f"{class_idx} {pts_str}")
            elif isinstance(seg, dict):
                polys = seg_to_polygons_from_rle(seg, height, width)
                if not polys:
                    print(f"RLE presente ma non decodificabile per annotation {ann.get('id')} (installa pycocotools per supporto RLE)")
                for poly in polys:
                    if len(poly) < 6:
                        continue
                    norm = normalize_poly(poly, width, height)
                    pts_str = ' '.join([f"{x:.6f}" for x in norm])
                    ann_lines.append(f"{class_idx} {pts_str}")
            else:
                # unknown segmentation format
                continue

        label_name = os.path.splitext(os.path.basename(fname))[0] + '.txt'
        label_path = os.path.join(dest_label_dir, label_name)
        write_label_file(label_path, ann_lines)

    # write data.yaml
    data_yaml_path = os.path.join(args.out_dir, 'data.yaml')
    rel_images_dir = 'images'
    with open(data_yaml_path, 'w', encoding='utf-8') as dy:
        dy.write(f"path: {args.out_dir}\n")
        dy.write(f"train: {rel_images_dir}/train\n")
        dy.write(f"val: {rel_images_dir}/val\n\n")
        dy.write("names:\n")
        for i, n in enumerate(names):
            dy.write(f"  {i}: {n}\n")

    print('Conversione completata. Dataset YOLO creato in:', args.out_dir)
    print('Contenuto principale: images/train images/val labels/train labels/val data.yaml')
    if mask_utils is None:
        print('Attenzione: pycocotools non trovato. Le maschere RLE non verranno decodificate.')

if __name__ == '__main__':
    main()
