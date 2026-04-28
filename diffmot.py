import os
import torch

import numpy as np
import os.path as osp
import logging
from torch import nn, optim, utils
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import cv2
from dataset import DiffMOTDataset
from models.autoencoder import D2MP
from models.condition_embedding import History_motion_embedding

import time
from tracker.tracker import KRAfTTracker
from tracker.basetrack import BaseTrack

from tracking_utils.log import logger
from tracking_utils.timer import Timer

def write_results(filename, results, data_type='mot'):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


class DiffMOT():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self.start_epoch = 1
        self._build()

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            pbar = tqdm(self.train_data_loader, ncols=80)
            for batch in pbar:
                for k in batch:
                    batch[k] = batch[k].to(device='cuda', non_blocking=True)

                train_loss = self.model(batch)
                train_loss = train_loss.mean()

                self.optimizer.zero_grad()
                pbar.set_description(f"Epoch {epoch},  Loss: {train_loss.item():.6f}")
                train_loss.backward()
                self.optimizer.step()

            if epoch % self.config.eval_every == 0:
                checkpoint = {
                    'ddpm': self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict()
                }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

    def eval(self, show=False, save_vis=False, save_video=True):
        det_root = self.config.det_dir
        img_root = getattr(self.config, "img_root", None)
        if not img_root:
            # Keep backward compatibility with old configs.
            img_root = getattr(self.config, "info_dir", "/cluster/pixstor/madrias-lab/Hasibur/MOT20/test")

        if not osp.isdir(img_root):
            raise FileNotFoundError(f"img_root does not exist: {img_root}")
        if not osp.isdir(det_root):
            raise FileNotFoundError(f"det_dir does not exist: {det_root}")

        seqs = getattr(self.config, "seqs", None)
        if isinstance(seqs, str):
            seqs = [s.strip() for s in seqs.split(",") if s.strip()]
        if not seqs:
            seqs = [s for s in os.listdir(img_root) if osp.isdir(osp.join(img_root, s))]
            seqs.sort()

        # Optional seqmap support for exact benchmark ordering.
        seqmap_file = getattr(self.config, "seqmap_file", None)
        if seqmap_file and osp.isfile(seqmap_file):
            with open(seqmap_file, "r") as f:
                parsed = [ln.strip() for ln in f.readlines()]
            parsed = [ln for ln in parsed if ln and ln.lower() != "name"]
            if parsed:
                seqs = parsed

        # Keep only valid sequences for this run.
        valid = []
        for seq in seqs:
            det_path = osp.join(det_root, seq)
            img_path = osp.join(img_root, seq, "img1")
            info_path = osp.join(self.config.info_dir, seq, "seqinfo.ini")
            if not osp.isdir(det_path):
                logger.info(f"[skip] missing det folder: {det_path}")
                continue
            if not osp.isdir(img_path):
                logger.info(f"[skip] missing image folder: {img_path}")
                continue
            if not osp.isfile(info_path):
                logger.info(f"[skip] missing seqinfo: {info_path}")
                continue
            valid.append(seq)

        if not valid:
            raise RuntimeError(
                f"No valid sequences found. img_root={img_root}, det_root={det_root}, info_dir={self.config.info_dir}"
            )

        seqs = sorted(valid)
        print(seqs)

        # root folder for visual frames
        vis_root = osp.join(self.config.save_dir, "vis")
        if save_vis:
            mkdirs(vis_root)

        # root folder for videos
        video_root = osp.join(self.config.save_dir, "videos")
        if save_video:
            mkdirs(video_root)

        stop_all = False

        for seq in seqs:
            if stop_all:
                break

            print(seq)
            det_path = osp.join(det_root, seq)
            img_path = osp.join(img_root, seq, "img1")

            info_path = osp.join(self.config.info_dir, seq, "seqinfo.ini")
            seq_info = open(info_path).read()
            seq_width = int(seq_info[seq_info.find("imWidth=") + 8:seq_info.find("\nimHeight")])
            seq_height = int(seq_info[seq_info.find("imHeight=") + 9:seq_info.find("\nimExt")])

            # try to parse frame rate from seqinfo if present
            fps = 25.0
            if "frameRate=" in seq_info:
                try:
                    fps_start = seq_info.find("frameRate=") + len("frameRate=")
                    fps_end = seq_info.find("\n", fps_start)
                    fps = float(seq_info[fps_start:fps_end].strip())
                except Exception:
                    fps = 25.0

            BaseTrack._count = 0
            tracker = KRAfTTracker(self.config)
            timer = Timer()
            results = []
            frame_id = 0

            frames = [s for s in os.listdir(det_path) if s.endswith(".txt")]
            frames.sort()
            image_files = [s for s in os.listdir(img_path)]
            image_files.sort()
            imgs = [f for f in image_files if not f.endswith(".txt")]

            num_steps = min(len(frames), len(imgs))
            if num_steps == 0:
                logger.info(f"[skip] empty sequence: {seq}")
                continue

            # folder for frames
            if save_vis:
                vis_seq_dir = osp.join(vis_root, seq)
                mkdirs(vis_seq_dir)

            # video writer for this sequence
            video_writer = None
            if save_video:
                first_img_path = osp.join(img_path, imgs[0])
                first_img = cv2.imread(first_img_path)
                h0, w0 = first_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = osp.join(video_root, f"{seq}_DiffMOT.mp4")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w0, h0))
                print(f"Writing video to {video_path}")

            # color map per id
            rng = np.random.RandomState(0)
            id_colors = {}

            for i in range(num_steps):
                f = frames[i]
                if stop_all:
                    break

                if frame_id % 10 == 0:
                    logger.info(
                        "Processing frame {} ({:.2f} fps)".format(
                            frame_id, 1.0 / max(1e-5, timer.average_time)
                        )
                    )

                timer.tic()
                f_path = osp.join(det_path, f)
                dets = np.loadtxt(f_path, dtype=np.float32, delimiter=",").reshape(-1, 6)[:, 1:6]

                im_path = osp.join(img_path, imgs[i])
                img = cv2.imread(im_path)
                tag = f"{seq}:{frame_id + 1}"

                # tracker update
                online_targets = tracker.update(
                            dets_xywhs=dets,
                            model=self.model,
                            img_w=seq_width,
                            img_h=seq_height,
                            tag=tag,
                            img=img,
                            dets_xywhs_95=None,   # or pass real backup if you have it
                        )

                online_tlwhs = []
                online_ids = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    if tid not in id_colors:
                        id_colors[tid] = tuple(int(c) for c in rng.randint(0, 255, size=3))

                    x1, y1, w, h = tlwh
                    x1i, y1i = int(x1), int(y1)
                    x2i, y2i = int(x1 + w), int(y1 + h)

                    cv2.rectangle(img, (x1i, y1i), (x2i, y2i), id_colors[tid], 2)
                    cv2.putText(
                        img,
                        f"{tid}",
                        (x1i, max(0, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        id_colors[tid],
                        2,
                        cv2.LINE_AA,
                    )

                timer.toc()

                # keep results for MOT txt
                results.append((frame_id + 1, online_tlwhs, online_ids))

                # overlay tracker name and fps
                current_fps = 1.0 / max(1e-5, timer.average_time)
                cv2.putText(
                    img,
                    "DiffMOT",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"FPS {current_fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # save frame image
                if save_vis:
                    out_img_name = f"{frame_id + 1:06d}.jpg"
                    out_img_path = osp.join(vis_seq_dir, out_img_name)
                    cv2.imwrite(out_img_path, img)

                # write to video
                if save_video and video_writer is not None:
                    video_writer.write(img)

                # optional live display
                if show:
                    cv2.imshow("KRAfT tracking", img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        stop_all = True
                        break

                frame_id += 1

            # tracker.dump_cache()

            if save_video and video_writer is not None:
                video_writer.release()

            # save MOT result file
            result_root = self.config.save_dir
            mkdirs(result_root)
            result_filename = osp.join(result_root, "{}.txt".format(seq))
            write_results(result_filename, results)

        if show:
            cv2.destroyAllWindows()


    def _build(self):
        self._build_dir()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_optimizer()

        # --- MINIMAL RESUME LOGIC (no scheduler needed) ---
        if getattr(self.config, "resume", False) and not self.config.eval_mode:
            assert self.config.resume_path, "Set config.resume_path to a saved checkpoint"
            print(f"> Resuming from: {self.config.resume_path}")
            ckpt = torch.load(self.config.resume_path, map_location="cpu")

            # 1) load model (strip DP prefix if present)
            state = ckpt.get('ddpm', ckpt)  # allow raw state_dict too
            target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            target.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=False)

            # 2) load optimizer if present (safe to skip if missing)
            if 'optimizer' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                except Exception as e:
                    print(f"[warn] couldn't load optimizer state: {e}")

            # 3) set next epoch
            self.start_epoch = int(ckpt.get('epoch', 0)) + 1
            print(f"> Will start training at epoch {self.start_epoch}")


            print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.eval_expname)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")


        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(checkpoint_dir, map_location = "cpu")

        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder(self):
        self.encoder = History_motion_embedding()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = D2MP(config, encoder=self.encoder)

        self.model = model
        if not self.config.eval_mode:
            self.model = torch.nn.DataParallel(self.model, self.config.gpus).to('cuda')
        else:
            self.model = self.model.cuda()
            self.model = self.model.eval()

        if self.config.eval_mode:
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in self.checkpoint['ddpm'].items()})

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        data_path = config.data_dir
        
        self.train_dataset = DiffMOTDataset(data_path, config)

        self.train_data_loader = utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.preprocess_workers,
            pin_memory=True
        )

    print("> Train Dataset built!")
