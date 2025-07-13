# How to install mmcv on Python 3.12

**mmcv** doesn't support Python 3.12+, so it is recommended to use Python 3.11 - see https://github.com/open-mmlab/mmcv/issues/3263

But if needed you can try to build mmcv for python 3.12 from source code:

```
git clone --branch v2.1.0 https://github.com/open-mmlab/mmcv
cd mmcv

# delete line "-r requirements/test.txt" from mmcv/requirements.txt - because onnxoptimizer can't be installed on Python 3.12

pip install -r requirements.txt

set MMCV_WITH_OPS=1
$Env:MMCV_WITH_OPS = 1
python setup.py develop
```

# Installation

Install torch with CUDA support:
```
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:
```
# Create virtual env if needed, then install package via setup.py.
# It will:
# - install openmim
# - install mmcv, mmdet
# - if Python 3.11 - download pre-trained weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
pip install -e .
```

[Download dataset](https://disk.yandex.ru/d/fdXRLa2Ju3Nflw) with partial labeling ([LabelImg](https://github.com/HumanSignal/labelImg) was used) and unzip it files into ```data/dataset01```

[Download pretrained](https://disk.yandex.ru/d/0fKV5m8In2YI4w) model weights and put it into pretrained directory. On Python 3.11 these weights were downloaded automatically via calling ```mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest pretrained``` (but this requires ```pip install openmim```, and it **fails** on Python 3.12, while works on Python 3.11).

# Usage

```
# generate dataset from manual labels
python src/create_tiles_dataset.py --input_dir data/dataset01 --output_dir data/dataset01_tiles640

# run training on data/dataset01_tiles640 with frozen backbone (i.e. fine-tune)
python src/train_rtmdet.py

python src/export_onnx.py --img data/dataset01/P1000101_1400494888594.JPG --ckpt models/ --out models/exported_model.onnx
```
