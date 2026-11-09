# Metrics

Every score comes from the metric authors' own code, kept unmodified in `third_party/`. `score_folder.m`, `score_all.py` and `score_uranker.py` only loop over images and save the scores.

| Metric | Source | Commit |
|---|---|---|
| UCIQE | https://github.com/JOU-UIP/UCIQE | `02a3cf3` |
| CEIQ | https://github.com/imfing/CEIQ | `c92b6a1` |
| FDUM | https://github.com/rmcong/Code-for-FDUM-method | `bd5d144` |
| CCF | https://github.com/zhenglab/CCF | `89cb931` |
| URanker | https://github.com/RQ-Wu/UnderwaterRanker | `f9bf350` |

FDUM's code ships as a zip, unpacked to `third_party/FDUM/extracted/`. URanker's weights come from [the authors' Google Drive](https://drive.google.com/drive/folders/1foW7uXG0GfdGzIDQEhXA82PJryGRQ3aM) and go in `third_party/UnderwaterRanker/checkpoints/URanker_ckpt.pth`.

## Requirements

URanker runs in Python, and uses a GPU if `torch` finds one. The other metrics need MATLAB with these toolboxes (tested on R2026b):

- Image Processing Toolbox: all metrics
- Statistics and Machine Learning Toolbox: FDUM (`skewness`) and CCF (`nanvar`, `nanmean`)
- Deep Learning Toolbox: CCF (`mapminmax`)

`score_all.py` expects MATLAB at the path in its `MATLAB` constant.

## Checks

| Metric | Check | Result |
|---|---|---|
| CEIQ | The three values in its README | Exact |
| FDUM | `Value/value.mat` shipped with the code | Exact |
| UCIQE, FDUM, CEIQ | The paper's raw row on UIEB and C60 | Exact |
| CCF | The paper's raw row | Within 0.1% |
| URanker | The paper's raw row | 0.3600 exact on UIEB; −0.3182 vs −0.3184 on C60 |

## Notes

- CEIQ runs the prebuilt `svmpredict.mexw64` from its repo, and FDUM runs the compiled `UIQM.p`. Neither can be read, so they run as supplied.
- The FDUM code is for non-commercial use only and asks that its paper be cited. CEIQ is MIT licensed. URanker is CC BY-NC 4.0, non-commercial only. The CCF and UCIQE repos state no licence.
