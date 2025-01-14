# TensoRF: Tensorial Radiance Fields

- This template uses the AI-Scientist to improve on the [TensoRF](https://arxiv.org/abs/2203.09517) paper.

- To run this, download synthetic nerf datasets from [here](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset/data) and place in datasets folder, such that the directory structure follows: 
```
datasets/
│
└── nerf_synthetic/
    │
    ├── chair/
    │   ├── train/
    │   ├── test/
    │   └── val/
    │
    ├── drums/
    │   ├── train/
    │   ├── test/
    │   └── val/
```

- To run on your own dataset, follow `configs/base_config.txt` to create your own config file.

- Code is largely sourced from official github repository (https://github.com/apchenstu/TensoRF) of TensoRF.

