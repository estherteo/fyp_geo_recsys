import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class YelpDataset(Dataset):
    def __init__(
            self,
            df_fpath: str,
            dataset_metadata: dict,
            model_type: str,
            threshold: int = 3
    ):
        self.df = pd.read_parquet(df_fpath)
        # Drop unusued columns
        #unused_cols = ['business_city', 'business_postal_code', 'business_text', 'review_bboxes']
        #self.df = self.df.drop(columns=unused_cols)

        # Drop Nans
        #self.df = self.df.dropna()

        # Convert user yelping since:
        """self.df['user_yelping_since'] = self.df['user_yelping_since'].str.replace('-',
                                                                    '')  # Remove the '-' from the date to convert to int
        self.df = self.df.astype(
            {'user_yelping_since': int}  # Convert the data from string to int
        )"""

        allowed_models = ['GMF', 'VanillaMLP', 'NeuMF', 'MLP', 'Frankie', 'MLPInt']
        assert model_type in allowed_models, f"Model type must be one of {allowed_models}"
        self.model_type = model_type
        self.dataset_metadata = dataset_metadata
        python_type_map_to_torch = {
            'uint8': torch.int,
            'uint16': torch.int,
            'uint32': torch.int,
            'int16': torch.int,
            'int64': torch.int,
            'float16': torch.float,
            'float64': torch.float
        }
        if self.df['review_stars'].max() == 1:
            self.df['review_stars'] *= 5

        if threshold is not None:
            if threshold is False:
                # Normalize Rating to [0, 1]
                self.df['review_stars'] = self.df['review_stars'].apply(lambda x: x / 5)
            else:
                # Binarize Rating
                self.df['review_stars'] = self.df['review_stars'].apply(lambda x: 1 if x >= threshold else 0)

        print("Dataset Metadata:", self.dataset_metadata['dtypes_map'])

        self.dataset_metadata['dtypes_map']['review_stars'] = 'float16'

        self.mapped_dtypes = {
            col: python_type_map_to_torch[self.dataset_metadata['dtypes_map'][col]] for col in self.df.columns
        }

        if self.model_type == 'GMF' or self.model_type == "VanillaMLP" or self.model_type == "NeuMF":
            self.df = self.df[['user_id', 'business_id', 'review_stars']]
            self.df = self.df.drop_duplicates()
        else:
            cols_to_remove = list({'review_date', 'business_is_open', 'review_id'}.intersection(set(self.df.columns)))
            self.df = self.df.drop(columns=cols_to_remove)

    def get_user_dim(self):
        return len([col for col in self.df.columns if col.startswith('user_') and col != 'user_id'])

    def get_item_dim(self):
        return len([col for col in self.df.columns if col.startswith('business_') and col != 'business_id'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.model_type == 'GMF' or self.model_type == "VanillaMLP" or self.model_type == "NeuMF":
            return {
                "user_id": torch.tensor(row["user_id"], dtype=torch.int),
                "business_id": torch.tensor(row["business_id"], dtype=torch.int),
                "review_stars": torch.tensor(row["review_stars"], dtype=torch.float),
            }
        else: # MLP or Frankie
            return {
                col: torch.tensor(row[col], dtype=self.mapped_dtypes[col]) for col in self.df.columns
            }


class YelpDataModule(pl.LightningDataModule):
    def __init__(self, train_ds_fpath: str, val_ds_fpath: str, test_ds_fpath: str,
                 dataset_metadata_fpath: str,
                 model_type: str,
                 train: bool,
                 threshold: int,
                 batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size
        self.train_fpath = train_ds_fpath
        self.val_fpath = val_ds_fpath
        self.test_fpath = test_ds_fpath

        # Load dataset metadata from pickled file in data dir
        self.dataset_metadata = pickle.load(open(dataset_metadata_fpath, 'rb'))
        self.dataset_metadata['dtypes_map']['review_stars'] = 'int16'
        self.model_type = model_type
        self.train = train
        self.threshold = threshold

    def setup(self, stage: str):
        if self.train:
            self.train_dataset = YelpDataset(
                df_fpath=self.train_fpath,
                dataset_metadata=self.dataset_metadata,
                model_type=self.model_type,
                threshold=self.threshold
            )

            self.val_dataset = YelpDataset(
                df_fpath=self.val_fpath,
                dataset_metadata=self.dataset_metadata,
                model_type=self.model_type,
                threshold=self.threshold
            )
        else:
            self.test_dataset = YelpDataset(
                df_fpath=self.test_fpath,
                dataset_metadata=self.dataset_metadata,
                model_type=self.model_type,
                threshold=self.threshold
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=5, shuffle=True,
                          persistent_workers=False, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=5, shuffle=False,
                          persistent_workers=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=5, shuffle=False,
                          persistent_workers=False, pin_memory=True)