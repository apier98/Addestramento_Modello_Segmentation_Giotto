"""
Run segmentation inference (YOLO v2.6 segmentation) on a video or webcam and save annotated output.
Usage examples:
  python detect_video_seg.py --source input.mp4 --weights runs\yolo26n-seg-run\weights\best.pt --out out.mp4
  python detect_video_seg.py --source 0 --weights runs\yolo26n-seg-run\weights\best.pt --out out.mp4  # webcam

The script uses ultralytics.YOLO and OpenCV. It attempts to read class names from the data.yaml referenced in runs/*/args.yaml if available.
"""

import argparse
import os
import re
import sys
import time

try:
    import cv2
except Exception as e:
    print('Please install opencv-python: pip install opencv-python')
    raise

try:
    from ultralytics import YOLO
except Exception:
    print('Please install ultralytics (pip install ultralytics>=2.6)')
    raise


def load_names_from_data_yaml(path):
    """Try to load names mapping from a data.yaml. Returns list of names or None."""
    if not os.path.isfile(path):
        return None
    # try yaml first
    try:
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        names = d.get('names') if isinstance(d, dict) else None
        if names is None:
            return None
        # names can be dict or list
        if isinstance(names, dict):
            # convert dict keyed by index to list
            sorted_items = sorted(names.items(), key=lambda x: int(x[0]))
            return [v for k, v in sorted_items]
        elif isinstance(names, (list, tuple)):
            return list(names)
    except Exception:
        # fallback naive parser
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            m = re.search(r"names:\s*(?:\n|\r\n)((?:\s+\d+:.*\n?)*)", content)
            if m:
                block = m.group(1).splitlines()
                # parse lines like '  0: Component_Base'
                names = []
                for line in block:
                    mm = re.search(r"\d+:\s*(.*)", line)
                    if mm:
                        names.append(mm.group(1).strip())
                if names:
                    return names
        except Exception:
            return None
    return None


def infer_video(source, weights, out_path=None, imgsz=640, conf=0.25, device=None, show=False):
    # prepare source for cv2 probing (to obtain fps/size)
    src_for_probe = source
    is_numeric = False
    if isinstance(source, str) and source.isdigit():
        src_for_probe = int(source)
        is_numeric = True

    cap = cv2.VideoCapture(src_for_probe)
    if not cap.isOpened():
        print(f'Warning: unable to open source for probing: {source}. Video writer properties will use defaults.')
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None
    cap.release()

    model = YOLO(weights)

    # create output writer if requested
    writer = None
    if out_path:
        # we'll determine frame size on first plotted frame if unknown
        writer = None

    print(f'Model: {weights}\nSource: {source}\nOutput: {out_path}\nimgsz: {imgsz}, conf: {conf}, device: {device or "auto"}')

    try:
        stream = model.predict(source=source, imgsz=imgsz, conf=conf, device=device, stream=True)
    except Exception as e:
        print('Model prediction failed:', e)
        return

    first = True
    frame_count = 0
    t0 = time.time()
    for result in stream:
        # result is a Results object
        try:
            # plot returns annotated image (numpy BGR)
            annotated = result.plot()  # BGR
        except Exception:
            # fallback: try to get original image
            annotated = getattr(result, 'orig_img', None)
            if annotated is None:
                continue

        if writer is None and out_path:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # ensure output directory exists
            os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
            writer = cv2.VideoWriter(out_path, fourcc, fps or 30.0, (w, h))

        if writer is not None:
            writer.write(annotated)

        if show:
            cv2.imshow('inference', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - t0
            print(f'Processed {frame_count} frames, avg FPS: {frame_count/elapsed:.2f}')

    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    total_time = time.time() - t0
    print(f'Done. Processed {frame_count} frames in {total_time:.2f}s ({(frame_count/total_time) if total_time>0 else 0:.2f} FPS)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run YOLO segmentation on video/webcam and save overlays')
    p.add_argument('--source', '-s', required=True, help='Video file path or webcam index (0)')
    p.add_argument('--weights', '-w', default='runs\\yolo26n-seg-run\\weights\\best.pt', help='Path to weights .pt')
    p.add_argument('--out', '-o', default=None, help='Output video path (e.g. out.mp4). If not provided, not saved.')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default=None, help='cuda or cpu (default auto)')
    p.add_argument('--show', action='store_true', help='Show live window')
    p.add_argument('--data', default=None, help='Optional path to data.yaml to read class names')
    args = p.parse_args()

    # attempt to find data.yaml referenced in runs args if not provided
    data_yaml = args.data
    if data_yaml is None:
        # try to locate runs/*/args.yaml
        runs_dir = os.path.join(os.getcwd(), 'runs')
        if os.path.isdir(runs_dir):
            for sub in os.listdir(runs_dir):
                candidate = os.path.join(runs_dir, sub, 'args.yaml')
                if os.path.isfile(candidate):
                    try:
                        with open(candidate, 'r', encoding='utf-8') as f:
                            txt = f.read()
                        m = re.search(r"data:\s*(.*)", txt)
                        if m:
                            candidate_data = m.group(1).strip()
                            # make path relative to project
                            candidate_data = os.path.normpath(os.path.join(os.getcwd(), candidate_data))
                            if os.path.isfile(candidate_data):
                                data_yaml = candidate_data
                                break
                    except Exception:
                        pass

    if data_yaml and os.path.isfile(data_yaml):
        names = load_names_from_data_yaml(data_yaml)
        if names:
            print('Loaded class names from', data_yaml)
            for i, n in enumerate(names):
                print(f'{i}: {n}')
        else:
            print('No class names found in', data_yaml)
    else:
        print('No data.yaml found or provided. Continuing without class names.')

    infer_video(args.source, args.weights, out_path=args.out, imgsz=args.imgsz, conf=args.conf, device=args.device, show=args.show)
