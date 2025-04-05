# Geo-Relevant Recommender System
## Setup
1. Deploy `pytorchlightning/pytorch_lightning` from [https://hub.docker.com/r/pytorchlightning/pytorch_lightning](https://hub.docker.com/r/pytorchlightning/pytorch_lightning).
2. Export environmental variables
   - export WANDB=...
   - export HF_WRITE_TOKEN=...
   - export HF_READ_TOKEN=...
   - export HF_HUB_ENABLE_HF_TRANSFER=1
   - export PYTHONPATH=... <br>
    **Note**: Set PYTHONPATH to the filepath of the parent folder of this repository.
4. Run `setup.sh`

<br>

## Pre-process Datasets
1. Download `estieeee/yelp2018_raw` from [huggingface](https://huggingface.co/datasets/estieeee/yelp2018_raw)
2. Run all cells of the `EDA_yelp2018.ipynb` from the repository.
3. At the end of pre-processing, take note of the filepaths of `train.parquet`, `val.parquet`, `test.parquet` and `dataset_metadata.pkl`.

<br>

## Train Baselines
There are two options -  training from scratch or loading from huggingface repo.
### Train from Scratch
1. In `model/baslines/baseline_train.py`, set the corresponding file paths in `_config`. Namely, set `train_ds_fpath` to the filepath of `train.parquet`, `val_ds_fpath` to the filepath of `val.parquet` and `test_ds_fpath` to the filepath of `test.parquet`. Also, set the value of key `dataset_metadata_fpath` in `_config` to the filepath of `dataset_metadata.pkl`.
2. In `model/baslines/baseline_train.py`, set the `preds_outpath` in the function `generate_predictions()` to a directory of your choice. Make sure it exists!
3. In `model/baslines/baseline_train.py`, set the `pred_df_fpath` in the function `get_metrics()` to the same directory as `pred_df_fpath` of `generate_predictions()`. Additionally, set `true_df_fpath` of `get_metrics()` to the filepath of `test.parquet` as generated in the pre-processing step.
4. Run the `model/baslines/baseline_train.py` script using:
   ```Python
   python  model/baslines/baseline_train.py
   ```
### Load from Huggingface Repo
1. Download `estieeee/yelp2018_baseline_models` from [huggingface](https://huggingface.co/estieeee/yelp2018_baseline_models)

<br>

## Train Wide-Deep
There are two options -  training from scratch or loading from huggingface repo.
### Train from Scratch
1. Set the `path` argument of `pd.read_parquet()` for `train`, `val` and `test` to that of the `train.parquet`, `val.parquet` and `test.parquet` files as derived from the pre-processing step.
2. Run the `model/wide_deep/wide_deep_train.py` script using:
   ```Python
   python  model/wide_deep/wide_deep_train.py
   ```
3. Repeat steps 1, 2 for `model/wide_deep/geo_wide_deep.py`
### Load from Huggingface Repo
1. Download `estieeee/yelp2018_models` from [huggingface](https://huggingface.co/estieeee/yelp2018_models).

<br>

## Evaluate Models
There are 2 options to evaluate the models - either score the models based on the predictions downloaded from the huggingface repo, or score the models by the predictions generated during the training step.
### Load from Huggingface Repo
#### Baselines
1. In the folder that you have downloaded `estieeee/yelp2018_baseline_models` to, locate `BaselineResults.ipynb`.
2. Run all cells for `BaselineResults.ipynb`.

#### WideDeep
1. In the folder that you ahve downloaded `estieeee/yelp2018_models` to, locate `WideDeepResults.ipynb`.
2. Run all cells for `WideDeepResults.ipynb`.

### Score trained models
#### Baselines
1. Set the `predictions_fpath` and `test_dataset_fpath` of the `load_results()` function in `BaselineResults.ipynb` to the predictions filepath (generated in the training step) as well as the `test.parquet` filepath (respectively). If you want to generate predictions post-filtered by the `min_bbox`, set the `geo_relevant_only` argument to `True`.
#### WideDeep
1. Set the `predictions_fpath` and `test_dataset_fpath` of the `load_results()` function in `WideDeepResults.ipynb` to the predictions filepath (generated in the training step) as well as the `test.parquet` filepath (respectively).

<br>

# Reproducibility Note
To further reproducibility of our results, as much as possible, we have uploaded all relevant artifacts of the project. In particular, both the raw and pre-processed dataset files have been uploaded to huggingface. Additionally, model weights along with their generated predictions have also been uploaded to huggingface, as were the functions we used to score the models. Please refer to the following table for the full list of huggingface repos:

| Resource               | Link                                                                                   | Notes                                                |
|------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------|
| Raw Dataset            | [estieeee/yelp2018_raw](https://huggingface.co/datasets/estieeee/yelp2018_raw)           | Contains the original, unmodified Yelp 2018 data along with the EDA notebook used to pre-process the data. |
| Pre-processed Dataset  | [estieeee/yelp2018_raw](https://huggingface.co/datasets/estieeee/yelp2018_raw)           | Cleaned and pre-processed version of the raw dataset that were used for training.|
| Baseline Models        | [estieeee/yelp2018_baseline_models](https://huggingface.co/estieeee/yelp2018_baseline_models) | Standard baseline model weights along with their predictions.   |
| WideDeep Models        | [estieeee/yelp2018_models](https://huggingface.co/estieeee/yelp2018_models)              | Wide and Deep models (with and without geo-relevant features) weights and their predictions.      |

To aid in reproducibility, the setup steps must be strictly followed. This means using the exact docker image as stated in step 1 of the setup section, and running the `setup.sh` script as stated in step 2 of the setup section. 
However, there are some caveats to reproducibility. Results are only fully reproducible when evaluation and scoring are carried out from the unaltered `BaselineResults.ipynb` and `WideDeepResults.ipynb` using the predictions downloaded from the huggingface repos listed in the above table. If model training is carried out and/or predictions are regenerated, slight deviations from our reported results are to be expected, due to the randomness inherent in any deep learning model training.

