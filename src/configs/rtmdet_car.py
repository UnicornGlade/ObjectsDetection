_base_ = '../../pretrained/rtmdet_tiny_8xb32-300e_coco.py'

num_classes = 1
class_names = ('car')

work_dir = 'models/rtmdet_car'

model = dict(
    bbox_head=dict(num_classes=num_classes),
)

img_scale = (640, 640)

TRAIN_SUBSET_SIZE = 1000
VAL_SUBSET_SIZE = 200

MAX_EPOCHS = 1000
VAL_EPOCHS_INTERVAL = 10

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         scale=[(640, 640), (800, 800)],
         keep_ratio=True,
         multiscale_mode='value'),
    dict(type='RandomAffine',
         max_rotate=0,
         scaling_ratio_range=(0.8, 1.2),
         max_translate_ratio=0.1,
         border=0,
         img_scale=img_scale),
    # alternative to YOLOXHSVRandomAug (without MMYOLO dependency)
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.8, 1.2),
         saturation_range=(0.8, 1.2),
         hue_delta=8),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=32,
    sampler=dict(
        type='RandomFixedCountSampler',
        subset_size=TRAIN_SUBSET_SIZE,
        shuffle=True,
        seed=0),
    dataset=dict(
        type='CocoDataset',
        data_root='data/dataset01_tiles640/',
        ann_file='instances_train.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=class_names),
    )
)

val_dataloader = dict(
    _delete_=True,                     # полностью перезаписываем
    batch_size=32,
    sampler=dict(
        type='RandomFixedCountSampler',
        subset_size=VAL_SUBSET_SIZE,
        shuffle=True,
        static_subset=True, # fix subset for all epochs
        seed=0),
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root='data/dataset01_tiles640/',   # ← ваша новая папка
        ann_file='instances_val.json',          # ← файл, который создал скрипт
        data_prefix=dict(img=''),               # картинки лежат прямо в корне
        test_mode=True,
        metainfo=dict(classes=('car')),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size=(640, 640),
                 pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape',
                            'img_shape', 'scale_factor')),
        ]))

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file='data/dataset01_tiles640/instances_val.json',      # тот же, что выше
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator

# ------------------------------------------------------------
# OPTIONAL: если у вас есть отдельный test-набор,
#           сделайте аналогичный блок test_dataloader.
#           Иначе можно просто:
test_dataloader = val_dataloader

# TODO minimize max_epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=MAX_EPOCHS, val_interval=VAL_EPOCHS_INTERVAL)

# Enable TensorBoard logging
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Configure logging hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)
