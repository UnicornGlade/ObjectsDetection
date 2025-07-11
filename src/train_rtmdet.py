"""Train RT‑DETR on the augmented dataset."""
import argparse, os, subprocess, sys, time, webbrowser, threading
from mmengine.config import Config
from mmengine.runner import Runner

def launch_tensorboard(log_dir, port=6006):
    """Launch TensorBoard in a separate process."""
    def run_tensorboard():
        try:
            # Check if TensorBoard is installed
            subprocess.run([sys.executable, "-c", "import tensorboard"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Installing TensorBoard...")
            subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"], check=True)
        
        # Launch TensorBoard
        cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port), "--reload_interval", "1"]
        print(f"Starting TensorBoard: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait a moment for TensorBoard to start
            time.sleep(3)
            
            # Open browser
            url = f"http://localhost:{port}"
            print(f"TensorBoard is running at: {url}")
            print("Opening TensorBoard in browser...")
            webbrowser.open(url)
            
            return process
            
        except Exception as e:
            print(f"Failed to start TensorBoard: {e}")
            return None
    
    # Run TensorBoard in a separate thread
    tensorboard_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tensorboard_thread.start()
    
    return tensorboard_thread

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='src/configs/rtmdet_car.py')
    ap.add_argument('--work_dir', default='models')
    ap.add_argument('--freeze_mode', choices=['backbone', 'partial', 'none'], default='backbone',
                    help='Layer freezing mode: backbone (freeze all except head), partial (intermediate freezing), none (full training)')
    ap.add_argument('--no-tensorboard', action='store_true',
                    help='Disable automatic TensorBoard launch')
    ap.add_argument('--tensorboard-port', type=int, default=6006,
                    help='Port for TensorBoard (default: 6006)')
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg)
    cfg.work_dir = args.work_dir
    cfg.resume = False
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Launch TensorBoard automatically unless disabled
    if not args.no_tensorboard:
        print("=" * 50)
        print("🚀 Launching TensorBoard automatically...")
        print("=" * 50)
        tensorboard_thread = launch_tensorboard(args.work_dir, args.tensorboard_port)
        time.sleep(2)  # Give TensorBoard a moment to start
    
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
    
    print("=" * 50)
    print("🎯 Starting RTMDet training...")
    if not args.no_tensorboard:
        print(f"📊 TensorBoard: http://localhost:{args.tensorboard_port}")
    print("=" * 50)
    
    runner.train()

if __name__ == '__main__':
    main()
