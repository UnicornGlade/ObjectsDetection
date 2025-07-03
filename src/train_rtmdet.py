"""Train RT‑DETR on the augmented dataset."""
import argparse, os
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='src/configs/rtmdet_car.py')
    ap.add_argument('--work_dir', default='models')
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = args.work_dir
    cfg.resume = False
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
