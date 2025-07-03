"""Export trained RT‑DETR checkpoint to ONNX."""        
import argparse, torch
from mmengine.config import Config
from mmdet.apis import init_detector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='configs/rtdetr_car.py')
    ap.add_argument('--ckpt', default='../models/rtdetr_car_last.pth')
    ap.add_argument('--out',  default='../models/rtdetr_car.onnx')
    args = ap.parse_args()

    model = init_detector(Config.fromfile(args.cfg), args.ckpt, device='cuda')
    model.eval()

    dummy = torch.randn(1, 3, 640, 640, device='cuda')
    torch.onnx.export(
        model,
        dummy,
        args.out,
        do_constant_folding=True,
        opset_version=12,
        input_names=['images'],
        output_names=['pred_logits', 'pred_boxes'],
        dynamic_axes={
            'images': {0: 'batch'},
            'pred_logits': {0: 'batch'},
            'pred_boxes': {0: 'batch'},
        },
    )
    print('Exported →', args.out)

if __name__ == '__main__':
    main()
