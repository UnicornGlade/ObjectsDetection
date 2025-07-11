from setuptools import setup, find_packages
import sys, subprocess

CUDA_TAG = 'cu121'  # change if you use another CUDA toolkit build

def _install_mmcv():
    # see https://github.com/open-mmlab/mmcv?tab=readme-ov-file#install-mmcv
    # pip install -U openmim
    # mim install mmcv==2.3.0
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'])
    subprocess.check_call([sys.executable, '-m', 'mim', 'install', 'mmcv==2.1.0'])

def _download_pretrained():
    # used in autolabel.py and in train_rtdetr.py
    # mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest pretrained
    subprocess.check_call([sys.executable, '-m', 'mim', 'download', 'mmdet', '--config', 'rtmdet_tiny_8xb32-300e_coco', '--dest', 'pretrained'])

_install_mmcv()
_download_pretrained()

setup(
    name='objects-detection',
    version='0.1.0',
    python_requires='==3.11',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        f'torch==2.3.0+{CUDA_TAG}',
        f'torchvision==0.18.0+{CUDA_TAG}',
        'mmengine>=0.10.3',
        'mmdet>=3.2.0',
        'albumentations>=1.4.0',
        'tensorboard>=2.19.0',
        'opencv-python>=4.10.0',
        'onnx>=1.16.0',
        'onnxruntime-gpu>=1.18.0',
        'tqdm',
        'numpy<2',
        'requests',
    ],
)
