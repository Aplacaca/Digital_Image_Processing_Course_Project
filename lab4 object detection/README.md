## SSD: Single Shot MultiBox Detector 

可视化：`python -m visdom.server`

训练：`python train.py`

评估：`python eval.py`

目录：

```bash
.
├── best.pth
├── dataset
│   └── data.py
│
├── detector.py
├── eval
│   ├── PR_50.jpg
│   └── mAP.jpg
│
├── eval.py
├── img.npy
├── loss.py
├── ssd.py
├── train.py
│
├── utils
│   ├── box_utils.py
│   ├── datetime_utils.py
│   ├── lr_schedule_utils.py
│   ├── prior_box.py
│   ├── random_seed.py
│   ├── visdom.py
│   └── visualize.py
│
└── vgg16_reducedfc.pth
```

