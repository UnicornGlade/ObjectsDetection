from setuptools import setup, find_packages
import sys, subprocess

CUDA_TAG = 'cu121'  # change if you use another CUDA toolkit build

def _install_torch():
    # pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.3.0+cu121', 'torchvision==0.18.0+cu121', '--extra-index-url', 'https://download.pytorch.org/whl/cu121'])

def _install_mmcv():
    # see https://github.com/open-mmlab/mmcv?tab=readme-ov-file#install-mmcv
    # pip install -U openmim
    # mim install mmcv==2.1.0
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'setuptools']) # to fix Python 3.12 - see https://github.com/open-mmlab/mmcv/issues/3263#issuecomment-2747343383
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv==2.1.0'])

def _install_mmdet():
    # we run it explicitly so that _download_pretrained() can proceed (it requires mmdet)
    # pip install mmengine>=0.10.3 mmdet>=3.2.0
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmengine>=0.10.3', 'mmdet>=3.2.0'])

def _download_pretrained():
    # used in autolabel.py and in train_rtdetr.py
    # mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest pretrained
    subprocess.check_call([sys.executable, '-m', 'mim', 'download', 'mmdet', '--config', 'rtmdet_tiny_8xb32-300e_coco', '--dest', 'pretrained'])

_install_torch()
_install_mmcv()
_install_mmdet()
_download_pretrained()

setup(
    name='objects-detection',
    version='0.1.0',
    python_requires='>=3.11',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        f'torch==2.3.0+{CUDA_TAG}',
        f'torchvision==0.18.0+{CUDA_TAG}',
        'albumentations>=1.4.0',
        'tensorboard>=2.19.0',
        'opencv-python>=4.10.0',
        'onnx>=1.16.0',
        'onnxruntime-gpu>=1.18.0',
        'tqdm',
        'numpy<2',
        'requests',
        'mmcv==2.1.0',
        'mmengine>=0.10.3',
        'mmdet>=3.2.0',
    ],
)
