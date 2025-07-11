"""Train RT‑DETR on the augmented dataset."""
import argparse, os
from mmengine.config import Config
from mmengine.runner import Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='src/configs/rtmdet_car.py')
    ap.add_argument('--work_dir', default='models')
    ap.add_argument('--freeze_mode', choices=['backbone', 'partial', 'none'], default='backbone',
                    help='Layer freezing mode: backbone (freeze all except head), partial (intermediate freezing), none (full training)')
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = args.work_dir
    cfg.resume = False
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    
    # Apply layer freezing based on mode
    if args.freeze_mode != 'none':
        model = runner.model
        for name, param in model.named_parameters():
            if args.freeze_mode == 'backbone' and 'backbone' in name:
                param.requires_grad = False
            elif args.freeze_mode == 'partial' and ('backbone' in name or 'neck' in name):
                param.requires_grad = False
            elif args.freeze_mode == 'backbone' and 'bbox_head' in name:
                param.requires_grad = True
            elif args.freeze_mode == 'partial' and 'bbox_head' in name:
                param.requires_grad = True
                
        print(f"Applied {args.freeze_mode} freeze mode. Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name} (trainable)")
    
    runner.train()

if __name__ == '__main__':
    main()
