"""
Track segmented components in a video using Ultralytics tracker and extract per-frame rectangular
measurements from segmentation masks. Saves per-frame CSV with one row per detection.

Usage example:
  python track_measure.py --source input.mp4 --weights runs\yolo26n-seg-run\weights\best.pt --out_csv out.csv

Requirements:
  pip install ultralytics opencv-python numpy

Notes:
 - The script prefers model.track(...) (Ultralytics tracker). If track IDs are not available,
   it falls back to per-frame generated IDs.
 - A detection is considered "touching_border" when its mask/bbox is within border_margin_px of the image edge.
 - Only frames where touching_border is False and area >= min_area_px are marked valid (but all rows are written with flag).
"""

import argparse
import os
import csv
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    raise RuntimeError('Please install ultralytics (pip install ultralytics>=2.6)')


def mask_to_numpy(mask_obj, H, W):
    """Robustly convert various ultralytics mask representations to a boolean numpy array (n, H, W).
    Uses the Ultralytics Results.Masks interface when available (masks.data or masks.numpy()).
    Returns None on failure.
    """
    if mask_obj is None:
        return None

    # 0) If mask_obj provides a numpy() convenience (BaseTensor), use it
    try:
        if hasattr(mask_obj, 'numpy'):
            try:
                mbase = mask_obj.numpy()
                # mbase may be a BaseTensor-like with .data attribute
                if hasattr(mbase, 'data'):
                    arr = np.asarray(mbase.data)
                else:
                    arr = np.asarray(mbase)
                arr = (arr > 0.5).astype(np.uint8)
                # normalize shape
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                # If shape is (H,W,N) or (H,W) transpose to (N,H,W)
                if arr.ndim == 3 and arr.shape[1] == H and arr.shape[2] == W and arr.shape[0] != mask_obj.shape()[0] if hasattr(mask_obj, 'shape') else True:
                    # uncertain ordering; try to detect and correct: if first dim equals H assume (H,W,N)
                    if arr.shape[0] == H:
                        arr = np.transpose(arr, (2, 0, 1))
                return arr
            except Exception:
                pass
    except Exception:
        pass

    # 1) direct data attribute (torch tensor or numpy) - common in many UL versions
    if hasattr(mask_obj, 'data'):
        try:
            m = mask_obj.data
            try:
                import torch
                if isinstance(m, torch.Tensor):
                    arr = m.cpu().numpy()
                else:
                    arr = np.asarray(m)
            except Exception:
                arr = np.asarray(m)
            arr = (arr > 0.5).astype(np.uint8)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            # ensure shape is (N,H,W)
            if arr.ndim == 3 and arr.shape[1] == H and arr.shape[2] == W and arr.shape[0] != len(arr):
                # if ordering looks wrong, attempt transpose
                if arr.shape[0] == H:
                    arr = np.transpose(arr, (2, 0, 1))
            return arr
        except Exception:
            pass

    # 2) polygon / xy formats (Masks.xy or Masks.xyn)
    polygons = None
    if hasattr(mask_obj, 'xy'):
        polygons = mask_obj.xy
    elif hasattr(mask_obj, 'xyn'):
        polygons = mask_obj.xyn

    if polygons is not None:
        masks = []
        for poly in polygons:
            # poly may be list of point tuples or flat list
            if poly is None:
                continue
            pts = None
            if isinstance(poly, (list, tuple)) and len(poly) > 0:
                if isinstance(poly[0], (list, tuple, np.ndarray)):
                    arr_coords = np.array(poly)
                else:
                    arr_coords = np.array(poly).reshape(-1, 2)
                # if normalized coords in [0,1]
                if arr_coords.max() <= 1.0:
                    arr_coords[:, 0] = arr_coords[:, 0] * W
                    arr_coords[:, 1] = arr_coords[:, 1] * H
                pts = arr_coords.astype(np.int32)
            if pts is None:
                continue
            m = np.zeros((H, W), dtype=np.uint8)
            try:
                cv2.fillPoly(m, [pts], 1)
            except Exception:
                try:
                    cv2.fillConvexPoly(m, pts, 1)
                except Exception:
                    continue
            masks.append(m)
        if masks:
            arr = np.stack(masks, axis=0)
            return arr.astype(np.uint8)

    # 3) COCO RLE / segmentation dict
    try:
        from pycocotools import mask as mask_utils
        if isinstance(mask_obj, dict) and 'counts' in mask_obj:
            decoded = mask_utils.decode(mask_obj)
            arr = np.asarray(decoded)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            return (arr > 0).astype(np.uint8)
        if isinstance(mask_obj, (list, tuple)) and len(mask_obj) > 0 and isinstance(mask_obj[0], dict) and 'counts' in mask_obj[0]:
            masks = [mask_utils.decode(m) for m in mask_obj]
            arr = np.stack(masks, axis=0)
            return (arr > 0).astype(np.uint8)
    except Exception:
        pass

    # 4) last-resort: try direct conversion
    try:
        arr = np.asarray(mask_obj)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return (arr > 0).astype(np.uint8)
    except Exception:
        return None


def extract_from_mask(mask):
    """Given 2D binary mask (uint8/bool), compute area, bbox, centroid, coords.
    Returns dict or None if empty.
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y_min = int(ys.min())
    y_max = int(ys.max())
    x_min = int(xs.min())
    x_max = int(xs.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = int(mask.sum())
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'width': width,
        'height': height,
        'area': area,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
    }


def try_get_track_ids(result):
    """Attempt to extract track ids from result.boxes or result.masks. Return list or None."""
    # result.boxes may have .id or .data
    try:
        if hasattr(result, 'boxes') and result.boxes is not None:
            # try .id attribute
            if hasattr(result.boxes, 'id'):
                ids = result.boxes.id
                try:
                    import torch
                    if isinstance(ids, torch.Tensor):
                        return ids.cpu().numpy().astype(int).tolist()
                except Exception:
                    return list(map(int, np.asarray(ids)))
            # some versions attach ids to boxes.data as last column; try heuristics
            if hasattr(result.boxes, 'data'):
                data = result.boxes.data
                arr = np.asarray(data)
                # shape (n, >=6) -> if last column seems integer and > 0 maybe id
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    possible_ids = arr[:, -1]
                    # if many non-zero integers and not fractional
                    if np.allclose(possible_ids, np.round(possible_ids)):
                        return list(map(int, possible_ids.tolist()))
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser(description='Track segmentation and extract per-frame rectangular measurements')
    p.add_argument('--source', '-s', required=True, help='Video file path or webcam index (0)')
    p.add_argument('--weights', '-w', default='runs/yolo26n-seg-run/weights/best.pt', help='Path to weights .pt')
    p.add_argument('--out_csv', default='results_track_measure.csv', help='Output CSV file path (will be overwritten)')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--device', default='cpu', help='cuda or cpu (default auto)')
    p.add_argument('--min_area_px', type=int, default=30, help='Minimum mask area in px to consider valid')
    p.add_argument('--min_frames', type=int, default=1, help='Minimum contiguous valid frames for an interval (post-processing)')
    p.add_argument('--border_margin_px', type=int, default=0, help='Pixels from image edge to consider touching border')
    p.add_argument('--show', action='store_true', help='Show live window with annotations (preview scaled to fit max display size)')
    p.add_argument('--display_max_width', type=int, default=1280, help='Maximum width for preview window (px). Frame is scaled for display only; processing remains full resolution.')
    p.add_argument('--display_max_height', type=int, default=720, help='Maximum height for preview window (px).')
    p.add_argument('--out_video', default=None, help='Optional annotated output video path')
    args = p.parse_args()

    # prepare video probe
    src_for_probe = args.source
    if isinstance(args.source, str) and args.source.isdigit():
        src_for_probe = int(args.source)
    cap = cv2.VideoCapture(src_for_probe)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()

    model = YOLO(args.weights)

    # Prepare CSV
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    csv_file = open(args.out_csv, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    header = ['video', 'frame_idx', 'time_s', 'track_id', 'class_id', 'class_name', 'bbox_x', 'bbox_y', 'width_px', 'height_px', 'area_px', 'centroid_x', 'centroid_y', 'rotated_width', 'rotated_height', 'rotated_angle_deg', 'rotated_box_pts', 'touching_border', 'confidence']
    writer.writerow(header)

    # prepare video writer if requested
    writer_vid = None
    total_frames = 0
    t0 = time.time()

    # run tracking if available (model.track), else fallback to predict
    use_track = True
    try:
        stream = model.track(source=args.source, imgsz=args.imgsz, conf=args.conf, device=args.device, stream=True, show=True, save=False)
    except Exception:
        use_track = False
        stream = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, device=args.device, stream=True, show=True, save=False)

    frame_idx = 0
    masks_detections = 0
    frames_with_masks = 0
    for result in stream:
        # result: ultralytics Results
        # Prefer Ultralytics' plotted overlay to match detect_video_seg.py visuals
        try:
            img = result.plot()  # annotated BGR image with masks/boxes/labels
        except Exception:
            img = getattr(result, 'orig_img', None)
            if img is None:
                continue
        H, W = img.shape[:2]

        # extract masks as numpy array (n, H, W) if possible
        masks_np = None
        if hasattr(result, 'masks') and result.masks is not None:
            masks_np = mask_to_numpy(result.masks, H, W)
            # diagnostic: if masks exist but conversion failed, dump attributes for inspection
            if (masks_np is None or masks_np.size==0):
                try:
                    dbg_dir = os.path.join('debug', 'mask_dumps')
                    os.makedirs(dbg_dir, exist_ok=True)
                    msg = []
                    msg.append(f'Frame {frame_idx}: result.masks type={type(result.masks)}')
                    try:
                        msg.append('attrs: ' + ','.join(dir(result.masks)))
                    except Exception:
                        pass
                    # try to inspect .data and .numpy and .xy/.xyn
                    try:
                        if hasattr(result.masks, 'data'):
                            md = getattr(result.masks, 'data')
                            msg.append(f'masks.data type={type(md)}, shape_attr={getattr(md, "shape", None)}')
                            try:
                                import torch
                                if isinstance(md, torch.Tensor):
                                    md_np = md.cpu().numpy()
                                    np.savez_compressed(os.path.join(dbg_dir, f'masks_frame_{frame_idx}_data.npz'), md=md_np)
                            except Exception:
                                try:
                                    md_np = np.asarray(md)
                                    np.savez_compressed(os.path.join(dbg_dir, f'masks_frame_{frame_idx}_data.npz'), md=md_np)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    try:
                        if hasattr(result.masks, 'numpy'):
                            try:
                                mn = result.masks.numpy()
                                # mn may be a BaseTensor-like
                                if hasattr(mn, 'data'):
                                    mn_arr = np.asarray(mn.data)
                                else:
                                    mn_arr = np.asarray(mn)
                                np.savez_compressed(os.path.join(dbg_dir, f'masks_frame_{frame_idx}_numpy.npz'), mn=mn_arr)
                                msg.append(f'masks.numpy() saved for frame {frame_idx}')
                            except Exception as e:
                                msg.append(f'masks.numpy() failed: {e}')
                    except Exception:
                        pass
                    try:
                        if hasattr(result.masks, 'xy'):
                            xy = result.masks.xy
                            # save first polygon
                            try:
                                np.savez_compressed(os.path.join(dbg_dir, f'masks_frame_{frame_idx}_xy.npz'), xy=np.array(xy, dtype=object))
                                msg.append('masks.xy saved')
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # write diagnostic text
                    with open(os.path.join(dbg_dir, f'masks_frame_{frame_idx}_debug.txt'), 'w', encoding='utf-8') as f:
                        f.write('\n'.join(msg))
                    print('Debug: masks present but conversion failed; dumped to', dbg_dir, 'frame', frame_idx)
                except Exception as e:
                    print('Debug dump failed:', e)
        if masks_np is not None and masks_np.size>0:
            masks_detections += masks_np.shape[0]
            frames_with_masks += 1
        # fallback: if no masks, attempt to create masks from boxes
        boxes_arr = None
        confs = []
        class_ids = []
        class_names = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            try:
                # Prefer attribute access (robust across ultralytics versions)
                # boxes coordinates
                if hasattr(result.boxes, 'xyxy'):
                    boxes_arr = np.asarray(result.boxes.xyxy).tolist()
                elif hasattr(result.boxes, 'xyxyn'):
                    boxes_arr = np.asarray(result.boxes.xyxyn).tolist()

                # confidences
                if hasattr(result.boxes, 'conf'):
                    try:
                        confs = np.asarray(result.boxes.conf).tolist()
                    except Exception:
                        confs = list(result.boxes.conf)

                # class ids
                if hasattr(result.boxes, 'cls'):
                    try:
                        class_ids = [int(x) for x in np.asarray(result.boxes.cls).tolist()]
                    except Exception:
                        class_ids = [int(x) for x in list(result.boxes.cls)]

                # fallback: inspect data matrix if some fields still missing
                data = getattr(result.boxes, 'data', None)
                if (boxes_arr is None or not confs or not class_ids) and data is not None:
                    arr = np.asarray(data)
                    if arr.ndim == 2 and arr.shape[1] >= 4:
                        boxes_arr = arr[:, :4].tolist()
                        if arr.shape[1] >= 6:
                            col4 = arr[:, 4]
                            col5 = arr[:, 5]
                            # heuristics: confidence is typically in [0,1]
                            if np.nanmax(col4) <= 1.0:
                                confs = col4.tolist()
                                class_ids = [int(x) for x in col5.tolist()]
                            elif np.nanmax(col5) <= 1.0:
                                confs = col5.tolist()
                                class_ids = [int(x) for x in col4.tolist()]
                            else:
                                # fallback: assume col4 is conf, col5 is class (best-effort)
                                confs = col4.tolist()
                                class_ids = [int(np.round(x)) for x in col5.tolist()]
            except Exception:
                boxes_arr = None
        # try to get track ids
        track_ids = try_get_track_ids(result)

        n_dets = 0
        if masks_np is not None:
            n_dets = masks_np.shape[0]
        elif boxes_arr is not None:
            n_dets = len(boxes_arr)

        # build default values lists
        if not confs:
            confs = [None] * n_dets
        if not class_ids:
            class_ids = [None] * n_dets
        # try to obtain class names from model if available
        model_names = getattr(model, 'names', None)
        if model_names and model_names != {}:
            class_names = [model_names.get(cid, '') if cid is not None else '' for cid in class_ids]
        else:
            class_names = ['' for _ in range(n_dets)]

        for det_i in range(n_dets):
            track_id = None
            if track_ids is not None and det_i < len(track_ids):
                try:
                    track_id = int(track_ids[det_i])
                except Exception:
                    track_id = track_ids[det_i]
            else:
                # fallback synthetic id
                track_id = f'{frame_idx}_{det_i}'

            conf = confs[det_i] if det_i < len(confs) else None
            class_id = class_ids[det_i] if det_i < len(class_ids) else None
            class_name = class_names[det_i] if det_i < len(class_names) else ''

            if masks_np is not None:
                mask = masks_np[det_i]
                info = extract_from_mask(mask)
                if info is None:
                    # empty mask, skip
                    continue
            else:
                # fallback: use bbox
                if boxes_arr is None:
                    continue
                x1, y1, x2, y2 = boxes_arr[det_i]
                x_min = int(max(0, np.floor(x1)))
                y_min = int(max(0, np.floor(y1)))
                x_max = int(min(W-1, np.ceil(x2)))
                y_max = int(min(H-1, np.ceil(y2)))
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                area = width * height
                centroid_x = x_min + width / 2.0
                centroid_y = y_min + height / 2.0
                info = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'width': width, 'height': height, 'area': int(area), 'centroid_x': centroid_x, 'centroid_y': centroid_y}

            touching = (info['x_min'] <= args.border_margin_px) or (info['y_min'] <= args.border_margin_px) or (info['x_max'] >= (W - 1 - args.border_margin_px)) or (info['y_max'] >= (H - 1 - args.border_margin_px))
            valid = (not touching) and (info['area'] >= args.min_area_px)

            # Compute oriented bounding box (minAreaRect) when mask is available
            rotated_w = ''
            rotated_h = ''
            angle_deg = ''
            box_pts_str = ''
            try:
                if masks_np is not None:
                    # prepare uint8 mask for contours
                    mask_uint8 = (mask.astype('uint8') * 255)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
                        (rcx, rcy), (rw, rh), rang = rect
                        rotated_w = float(rw)
                        rotated_h = float(rh)
                        angle_deg = float(rang)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        box_pts_str = ';'.join([f'{int(x)}_{int(y)}' for x, y in box])
                else:
                    # fallback: use axis-aligned bbox as rotated box
                    rotated_w = float(info['width'])
                    rotated_h = float(info['height'])
                    angle_deg = 0.0
                    x0 = info['x_min']; y0 = info['y_min']; x1 = info['x_max']; y1 = info['y_max']
                    box_pts_str = f'{x0}_{y0};{x1}_{y0};{x1}_{y1};{x0}_{y1}'
            except Exception:
                rotated_w = ''
                rotated_h = ''
                angle_deg = ''
                box_pts_str = ''

            time_s = frame_idx / (fps or 30.0)
            row = [args.source, frame_idx, f'{time_s:.4f}', track_id, class_id, class_name, info['x_min'], info['y_min'], info['width'], info['height'], info['area'], f'{info["centroid_x"]:.2f}', f'{info["centroid_y"]:.2f}', rotated_w, rotated_h, angle_deg, box_pts_str, int(touching), f'{conf:.4f}' if conf is not None else '']
            writer.writerow(row)

            # Draw ID and area text on the annotated image (avoid redrawing boxes to preserve original overlay)
            text = f'id:{track_id} a:{info["area"]}'
            tx = int(info['x_min'])
            ty = int(max(0, info['y_min'] - 6))
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # initialize video writer if needed
        if args.out_video and writer_vid is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(args.out_video) or '.', exist_ok=True)
            writer_vid = cv2.VideoWriter(args.out_video, fourcc, fps or 30.0, (w, h))
        if writer_vid is not None:
            writer_vid.write(img)

        if args.show:
            # scale frame for display while processing at full resolution
            max_w = args.display_max_width
            max_h = args.display_max_height
            scale = min(1.0, float(max_w) / float(W) if W>0 else 1.0, float(max_h) / float(H) if H>0 else 1.0)
            if scale < 1.0:
                disp_w = int(W * scale)
                disp_h = int(H * scale)
                disp_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                disp_img = img
            cv2.imshow('track_measure', disp_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        total_frames += 1
        if total_frames % 100 == 0:
            elapsed = time.time() - t0
            print(f'Processed {total_frames} frames, avg FPS: {total_frames/elapsed:.2f}')

    csv_file.close()
    if writer_vid:
        writer_vid.release()
    if args.show:
        cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f'Done. Wrote {args.out_csv} for {total_frames} frames in {elapsed:.2f}s')
    # diagnostics
    print(f'Mask diagnostics: frames_with_masks={frames_with_masks}, mask_detections={masks_detections}')


if __name__ == '__main__':
    main()
