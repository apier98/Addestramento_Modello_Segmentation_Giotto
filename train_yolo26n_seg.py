#!/usr/bin/env python3
"""
train_yolo26n_seg.py

Script per avviare l'addestramento di YOLO 2.6 (segmentazione) - variante nano (yolo26n-seg)
Usa l'API Python di Ultralytics.

Esempio:
python train_yolo26n_seg.py --data dataset_yolo/data.yaml --model yolo26n-seg.pt --epochs 100 --imgsz 640 --batch 16 --device 0 --project runs --name yolo26n-seg-run

Requisiti:
pip install ultralytics>=2.6

Il file accetta opzioni per percorso dei dati, numero di epoche, dimensione immagine, batch, device e altro.
"""

import argparse
import sys
import os

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics non trovato. Installa con: pip install ultralytics>=2.6\nErrore: " + str(e))


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLO 2.6 nano segmentation (yolo26n-seg)')
    p.add_argument('--data', required=True, help='Path to data.yaml (dataset root)')
    p.add_argument('--model', default='yolo26n-seg.pt', help='Model weights or model name (default: yolo26n-seg.pt)')
    p.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    p.add_argument('--imgsz', type=int, default=640, help='Image size (px)')
    p.add_argument('--batch', type=int, default=16, help='Batch size')
    p.add_argument('--device', default='0', help='Device id or cpu (e.g. 0 or cpu)')
    p.add_argument('--project', default='runs', help='Project folder to save runs')
    p.add_argument('--name', default='yolo26n-seg', help='Run name')
    p.add_argument('--exist_ok', action='store_true', help='Overwrite existing project/name')
    p.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    p.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    p.add_argument('--resume', action='store_true', help='Resume training from last checkpoint if available')
    p.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs (if supported)')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"data.yaml non trovato: {args.data}")

    print('Caricamento modello:', args.model)
    model = YOLO(args.model)

    print('Avvio training:')
    print(f"  data: {args.data}")
    print(f"  epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")
    print(f"  device: {args.device}, project: {args.project}, name: {args.name}")

    # Ultralytics train kwargs - these are commonly supported; additional args can be passed
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        workers=args.workers,
        lr0=args.lr0,
        resume=args.resume,
    )

    # For segmentation models the ultralytics API will handle mask loss/etc. No need to set task explicitly if model supports seg.
    try:
        model.train(**train_kwargs)
    except TypeError:
        # Fallback: some ultralytics versions expect slightly different kwargs; try a minimal call
        print('Warning: argomenti di training non accettati direttamente dalla versione di Ultralytics installata. Eseguo chiamata semplificata.')
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)

    print('Training completato (o terminato). Controlla la cartella', os.path.join(args.project, args.name))

if __name__ == '__main__':
    main()
