# How to install mmcv on Python 3.12 (currently doesn't work)

**mmcv** doesn't support Python 3.12+, so it is recommended to use Python 3.11 - see https://github.com/open-mmlab/mmcv/issues/3263

But if needed you can try to build mmcv for python 3.12 from source code:

```
git clone --branch v2.1.0 https://github.com/open-mmlab/mmcv
cd mmcv

# delete line "-r requirements/test.txt" from mmcv/requirements.txt - because onnxoptimizer can't be installed on Python 3.12

pip install -r requirements.txt

set MMCV_WITH_OPS=1
set FORCE_CUDA=1
python -m pip install -e .
```

But sadly I still wasn't able to get over this error on starting training:

<details>
Traceback (most recent call last):
  File "C:\...\src\train_rtmdet.py", line 100, in <module>
    main()
  File "C:\...\src\train_rtmdet.py", line 71, in main
    runner = Runner.from_cfg(cfg)
             ^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\runner\runner.py", line 462, in from_cfg
    runner = cls(
             ^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\runner\runner.py", line 397, in __init__
    self.log_processor = self.build_log_processor(log_processor)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\runner\runner.py", line 1650, in build_log_processor   
    log_processor = LOG_PROCESSORS.build(log_processor_cfg)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\registry\registry.py", line 570, in build
    return self.build_func(cfg, *args, **kwargs, registry=self)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\registry\build_functions.py", line 98, in build_from_cfg
    obj_cls = registry.get(obj_type)
              ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmengine\registry\registry.py", line 451, in get
    self.import_from_location()
  File "C:\...\.venv\Lib\site-packages\mmengine\registry\registry.py", line 376, in import_from_location
    import_module(loc)
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\importlib\__init__.py", line 90, in import_module   
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\...\.venv\Lib\site-packages\mmdet\engine\__init__.py", line 2, in <module>
    from .hooks import *  # noqa: F401, F403
    ^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmdet\engine\hooks\__init__.py", line 10, in <module>
    from .visualization_hook import (DetVisualizationHook,
  File "C:\...\.venv\Lib\site-packages\mmdet\engine\hooks\visualization_hook.py", line 14, in <module> 
    from mmdet.datasets.samplers import TrackImgSampler
  File "C:\...\.venv\Lib\site-packages\mmdet\datasets\__init__.py", line 31, in <module>
    from .utils import get_loading_pipeline
  File "C:\...\.venv\Lib\site-packages\mmdet\datasets\utils.py", line 5, in <module>
    from mmdet.datasets.transforms import LoadAnnotations, LoadPanopticAnnotations
  File "C:\...\.venv\Lib\site-packages\mmdet\datasets\transforms\__init__.py", line 6, in <module>     
    from .formatting import (ImageToTensor, PackDetInputs, PackReIDInputs,
  File "C:\...\.venv\Lib\site-packages\mmdet\datasets\transforms\formatting.py", line 11, in <module>  
    from mmdet.structures.bbox import BaseBoxes
  File "C:\...\.venv\Lib\site-packages\mmdet\structures\bbox\__init__.py", line 2, in <module>
    from .base_boxes import BaseBoxes
  File "C:\...\.venv\Lib\site-packages\mmdet\structures\bbox\base_boxes.py", line 9, in <module>       
    from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks
  File "C:\...\.venv\Lib\site-packages\mmdet\structures\mask\__init__.py", line 3, in <module>
    from .structures import (BaseInstanceMasks, BitmapMasks, PolygonMasks,
  File "C:\...\.venv\Lib\site-packages\mmdet\structures\mask\structures.py", line 12, in <module>      
    from mmcv.ops.roi_align import roi_align
  File "C:\...\.venv\Lib\site-packages\mmcv\ops\__init__.py", line 3, in <module>
    from .active_rotated_filter import active_rotated_filter
  File "C:\...\.venv\Lib\site-packages\mmcv\ops\active_rotated_filter.py", line 10, in <module>        
    ext_module = ext_loader.load_ext(
                 ^^^^^^^^^^^^^^^^^^^^
  File "C:\...\.venv\Lib\site-packages\mmcv\utils\ext_loader.py", line 13, in load_ext
    ext = importlib.import_module('mmcv.' + name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0\Lib\importlib\__init__.py", line 90, in import_module   
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'mmcv._ext'
</details>

# Installation

Install torch with CUDA support:
```
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:
```
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
