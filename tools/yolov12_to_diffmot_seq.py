import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO


# =========================================================
# Edit only this block if needed
# =========================================================
CONFIG = {
    # Dataset layout expected
    # data_root / dataset / mode / seq_name / img1
    "data_root": "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID/BoT-SORT/TrackEval/data/gt/mot_challenge/UAVSwarm",
    "dataset": "UAVSwarm-test",
    "mode": "",  # train or val or test or empty if your layout has no extra split folder

    # YOLOv12
    "yolo_weights": "/cluster/pixstor/madrias-lab/Hasibur/AT/YOLOv12-BoT-SORT-ReID/BoT-SORT/yolov12/runs/uav-swarm/12l/uavswarm/weights/best.pt",
    "conf": 0.01,

    # Output root for per frame txt
    # This script will create
    # /cluster/pixstor/madrias-lab/Hasibur/AT/ResDiffMOT-v2/yolov12det/0.80/seq_name/frame.txt
    # /cluster/pixstor/madrias-lab/Hasibur/AT/ResDiffMOT-v2/yolov12det/0.95/seq_name/frame.txt
    "out_txt_root": "/cluster/pixstor/madrias-lab/Hasibur/AT/ResDiffMOT-v3/yolov12detv5",

    # Two NMS IoU settings
    "iou_a": 0.60,
    "iou_b": 0.45,
}
# =========================================================


def ensure_sorted_frame_list(img_dir):
    imgs = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))
    imgs.sort()
    return imgs


def run_yolo_on_frame(model, img_bgr, conf, iou):
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        imgsz=1280,
        verbose=False
    )
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.detach().cpu().numpy().astype(np.float32)

    return xyxy, scores, cls


def save_frame_txt(txt_path, frame_id, xyxy, scores):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "w") as f:
        if xyxy is None or len(xyxy) == 0:
            return

        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            w = x2 - x1
            h = y2 - y1
            score = scores[j]

            # Same style as your earlier code, but using real frame_id
            # If you want the first field always 1, replace frame_id with 1
            line = f"{frame_id},{x1:.4f},{y1:.4f},{w:.4f},{h:.4f},{score:.4f}\n"
            f.write(line)


def process_sequence_to_txt(model, seq_dir, conf, iou_a, iou_b, out_root):
    img_dir = os.path.join(seq_dir, "img1")
    imgs = ensure_sorted_frame_list(img_dir)

    seq_name = os.path.basename(seq_dir)

    out_dir_a = os.path.join(out_root, "0.80", seq_name)
    out_dir_b = os.path.join(out_root, "0.95", seq_name)

    for idx, img_path in enumerate(imgs):
        frame_id = idx + 1
        img = cv2.imread(img_path)
        if img is None:
            continue

        frame_stem = os.path.splitext(os.path.basename(img_path))[0]

        # IoU 0.80
        xyxy_a, scores_a, _ = run_yolo_on_frame(model, img, conf, iou_a)
        txt_a = os.path.join(out_dir_a, f"{frame_stem}.txt")
        save_frame_txt(txt_a, frame_id, xyxy_a, scores_a)

        # IoU 0.95
        xyxy_b, scores_b, _ = run_yolo_on_frame(model, img, conf, iou_b)
        txt_b = os.path.join(out_dir_b, f"{frame_stem}.txt")
        save_frame_txt(txt_b, frame_id, xyxy_b, scores_b)


def main():
    cfg = CONFIG

    yolo = YOLO(cfg["yolo_weights"])

    split_dir = os.path.join(cfg["data_root"], cfg["dataset"], cfg["mode"])
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    seq_names = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])

    for seq_name in seq_names:
        seq_dir = os.path.join(split_dir, seq_name)
        print("Processing", seq_name)

        process_sequence_to_txt(
            model=yolo,
            seq_dir=seq_dir,
            conf=cfg["conf"],
            iou_a=cfg["iou_a"],
            iou_b=cfg["iou_b"],
            out_root=cfg["out_txt_root"]
        )

    print("Saved per frame txt under", os.path.abspath(cfg["out_txt_root"]))
    print("Folder for 0.80:", os.path.join(os.path.abspath(cfg["out_txt_root"]), "0.80"))
    print("Folder for 0.95:", os.path.join(os.path.abspath(cfg["out_txt_root"]), "0.95"))


if __name__ == "__main__":
    main()
