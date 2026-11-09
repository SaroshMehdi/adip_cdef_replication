# Data

Datasets are not committed. The code reads images from these folders:

```
data/
├── UIEB/
│   ├── raw/           890 images
│   ├── reference/     890 images, same file names as raw/
│   └── challenging/   60 images, the paper's "C60"
├── LOLv2/             Extensions 2 and 3
│   ├── Real_captured/{Train,Test}/
│   └── Synthetic/{Train,Test}/
├── EUVP/
│   ├── test_samples/Inp/   515 raw test images (256 x 256): the paper's EUVP set, confirmed by its raw scores
│   ├── test_samples/GTr/   515 references, same file names as Inp/
│   ├── Paired/, Unpaired/  training and validation sets
│   └── eval_data/          other methods' outputs shipped with the dataset (not raw images)
└── UCCS/              300 images (400 x 300), 100 each in blue/, green/, blue-green/
```

Images are processed at their original resolution, as in the paper.

| Dataset | Source |
|---|---|
| UIEB | Li et al., *IEEE TIP* 2020: https://li-chongyi.github.io/proj_benchmark.html |
| EUVP | Islam et al., *IEEE RA-L* 2020: https://irvlab.cs.umn.edu/resources/euvp-dataset. That page's Google Drive link no longer works, so this copy came from the Kaggle mirror https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset (CC0). Its image counts match the official statistics |
| UCCS | Liu et al., *IEEE TCSVT* 2020 (the RUIE benchmark): the `UCCS/` folder of https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark |
| LOL-v2 | Yang et al., *IEEE TIP* 2021 |
