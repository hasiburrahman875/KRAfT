import os
import glob
import re
import numpy as np
import cv2
from ultralytics import YOLO


# =========================================================
# Edit only this block if needed
# =========================================================
CONFIG = {
    # MOT root directory
    # Expected structure:
    # mot_root / split / seq_name / img1
    # mot_root / split / seq_name / det / det.txt
    "mot_root": "/cluster/pixstor/madrias-lab/Hasibur/MOT20",
    "split": "test",  # train or val or test

    # YOLO
    "yolo_weights": "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID/BoT-SORT/yolov12/runs/uav-swarm/12l/uavswarm/weights/best.pt",
    "conf": 0.01,
    "imgsz": 1280,

    # NMS IoU you want for exported det.txt
    "iou": 0.45,

    # Safety switch
    # If True, will overwrite existing det.txt
    "overwrite": True,
}
# =========================================================


def natural_key(path):
    base = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[-1])
    return base


def ensure_sorted_frame_list(img_dir):
    imgs = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    imgs.sort(key=natural_key)
    return imgs


def run_yolo_on_frame(model, img_bgr, conf, iou, imgsz):
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)

    return xyxy, scores


def xyxy_to_tlwh(xyxy):
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=1)


def process_sequence_and_write_det(model, seq_dir, conf, iou, imgsz, overwrite=True):
    """
    Reads seq_dir/img1 and writes seq_dir/det/det.txt in MOT format.
    Overwrites by default.
    """
    img_dir = os.path.join(seq_dir, "img1")
    if not os.path.isdir(img_dir):
        print("Skipping, img1 not found:", img_dir)
        return None

    imgs = ensure_sorted_frame_list(img_dir)
    if len(imgs) == 0:
        print("Skipping, no images:", img_dir)
        return None

    det_dir = os.path.join(seq_dir, "det")
    os.makedirs(det_dir, exist_ok=True)

    det_path = os.path.join(det_dir, "det.txt")

    if os.path.exists(det_path) and not overwrite:
        print("det.txt exists and overwrite=False, skipping:", det_path)
        return det_path

    det_lines = []

    for idx, img_path in enumerate(imgs):
        frame_id = idx + 1
        img = cv2.imread(img_path)
        if img is None:
            continue

        xyxy, scores = run_yolo_on_frame(model, img, conf, iou, imgsz)

        if xyxy is None or len(xyxy) == 0:
            continue

        tlwh = xyxy_to_tlwh(xyxy)

        for j in range(tlwh.shape[0]):
            x, y, w, h = tlwh[j]
            score = float(scores[j])

            # MOT detection format:
            # frame, id, x, y, w, h, score, -1, -1, -1
            det_lines.append(
                f"{frame_id},-1,{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.6f},-1,-1,-1\n"
            )

    with open(det_path, "w") as f:
        f.writelines(det_lines)

    return det_path


def list_sequences(split_dir):
    """
    Lists sequence directories under mot_root/split.
    A valid sequence directory should be a folder.
    """
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    seq_dirs = []
    for name in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, name)
        if os.path.isdir(d):
            seq_dirs.append(d)

    return seq_dirs


def main():
    cfg = CONFIG

    split_dir = os.path.join(cfg["mot_root"], cfg["split"])
    seq_dirs = list_sequences(split_dir)

    yolo = YOLO(cfg["yolo_weights"])

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        print("Processing", seq_name)

        det_path = process_sequence_and_write_det(
            model=yolo,
            seq_dir=seq_dir,
            conf=float(cfg["conf"]),
            iou=float(cfg["iou"]),
            imgsz=int(cfg["imgsz"]),
            overwrite=bool(cfg["overwrite"])
        )

        if det_path:
            print("Wrote", det_path)

    print("Done. Replaced det.txt files under", split_dir)


if __name__ == "__main__":
    main()
