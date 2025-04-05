import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model.baselines.dataloader import YelpDataModule
from model.baselines.generate_baseline_predictions import generate_predictions
from model.baselines.get_metrics import get_metrics
from model.baselines.layers.joint_gmf_model import JointGMF
import torch

from model.baselines.layers.joint_vanilla_mlp_model import JointVanillaMLP
from model.baselines.models.NeuMF import NeuMF

torch.set_float32_matmul_precision('high')


def _pretrain_gmf(config):
    datamodule = YelpDataModule(batch_size=config['batch_size'],
                                train_ds_fpath=config['train_ds_fpath'],
                                val_ds_fpath=config['val_ds_fpath'],
                                test_ds_fpath=config['test_ds_fpath'],
                                dataset_metadata_fpath=config['dataset_metadata_fpath'],
                                model_type='GMF',
                                threshold=config['RatingThreshold'],
                                train=True)

    gmf_checkpoint_callback = ModelCheckpoint(
        monitor='GMF_val_loss',
        dirpath='checkpoints',
        filename='gmf-{epoch:02d}.ckpt',
        save_top_k=1,
    )

    model = JointGMF(
        num_users=config['num_users'],
        num_items=config['num_items'],
        num_factors=config['num_factors'],
        rating_threshold=config['RatingThreshold']
    )
    #model = torch.compile(model)
    wandb_logger = WandbLogger(log_model=False)

    early_stop_callback = EarlyStopping(monitor="GMF_val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator=config['accelerator'],
                         devices=config['devices'],
                         log_every_n_steps=15,
                         callbacks=[gmf_checkpoint_callback, early_stop_callback], logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #strategy=config['strategy'],
                         precision="bf16-mixed")
    trainer.fit(model, datamodule=datamodule)

    return gmf_checkpoint_callback.best_model_path


def _pretrain_vanilla_mlp(config):
    datamodule = YelpDataModule(batch_size=config['batch_size'],
                                train_ds_fpath=config['train_ds_fpath'],
                                val_ds_fpath=config['val_ds_fpath'],
                                test_ds_fpath=config['test_ds_fpath'],
                                dataset_metadata_fpath=config['dataset_metadata_fpath'],
                                model_type='VanillaMLP',
                                threshold=config['RatingThreshold'],
                                train=True)

    mlp_checkpoint_callback = ModelCheckpoint(
        monitor='Vanilla_MLP_val_loss',
        dirpath='checkpoints',
        filename='vanilla_mlp-{epoch:02d}.ckpt',
        save_top_k=1,
    )

    model = JointVanillaMLP(
        num_users=config['num_users'],
        num_items=config['num_items'],
        num_factors=config['num_factors'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        rating_threshold=config['RatingThreshold'],
    )
    #model = torch.compile(model)
    wandb_logger = WandbLogger(log_model=False)

    early_stop_callback = EarlyStopping(monitor="Vanilla_MLP_val_loss", min_delta=0.001, patience=3, verbose=False,
                                        mode="min")

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator=config['accelerator'],
                         devices=config['devices'],
                         callbacks=[mlp_checkpoint_callback, early_stop_callback],
                         log_every_n_steps=5,
                         precision="bf16-mixed",
                         check_val_every_n_epoch=1,
                         #strategy=config['strategy'],
                         logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)

    return mlp_checkpoint_callback.best_model_path


def _pretrain_neumf(config):
    datamodule = YelpDataModule(batch_size=config['batch_size'],
                                train_ds_fpath=config['train_ds_fpath'],
                                val_ds_fpath=config['val_ds_fpath'],
                                test_ds_fpath=config['test_ds_fpath'],
                                dataset_metadata_fpath=config['dataset_metadata_fpath'],
                                model_type='NeuMF',
                                threshold=config['RatingThreshold'],
                                train=True)

    neumf_checkpoint_callback = ModelCheckpoint(
        monitor='NeuMF_val_loss',
        dirpath='checkpoints',
        filename='NeuMF-{epoch:02d}.ckpt',
        save_top_k=1,
    )

    gmf = JointGMF.load_from_checkpoint(
        config['gmf_checkpoint_path'],
        num_users=config['num_users'],
        num_items=config['num_items'],
        num_factors=config['num_factors'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        rating_threshold=config['RatingThreshold'],
    )
    mlp = JointVanillaMLP.load_from_checkpoint(
        config['vanilla_mlp_checkpoint_path'],
        num_users=config['num_users'],
        num_items=config['num_items'],
        num_factors=config['num_factors'],
        num_layers=config['num_layers'],
        rating_threshold=config['RatingThreshold'],
        dropout=config['dropout'],
    )

    model = NeuMF(
        num_users=config['num_users'],
        num_items=config['num_items'],
        rating_threshold=config['RatingThreshold'],
        num_factors=config['num_factors'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        GMF_model=gmf,
        MLP_model=mlp
    )
    wandb_logger = WandbLogger(log_model=False)

    early_stop_callback = EarlyStopping(monitor="NeuMF_val_loss", min_delta=0.001, patience=3, verbose=False,
                                        mode="min")

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator=config['accelerator'],
                         devices=config['devices'],
                         callbacks=[neumf_checkpoint_callback, early_stop_callback],
                         log_every_n_steps=5,
                         precision="bf16-mixed",
                         check_val_every_n_epoch=1,
                         # strategy=config['strategy'],
                         logger=wandb_logger)
    trainer.fit(model, datamodule=datamodule)

    return neumf_checkpoint_callback.best_model_path

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    _config = {
        #'strategy': 'ddp',
        "accelerator": 'gpu',
        'devices': 1,
        "batch_size": 8192,
        "train_ds_fpath": "/root/data/train.parquet",
        "val_ds_fpath": "/root/data/val.parquet",
        "test_ds_fpath": "/root/data/test.parquet",
        "dataset_metadata_fpath": "/root/data/dataset_metadata.pkl",
        "num_users": 575174,
        "num_items": 44575,
        'num_user_dim': 3,
        'num_item_dim': 2,
        "max_epochs": 100,
        'RatingThreshold': 3, # At and above which is 1, below is 0
        'num_int_dim': 4,
        'final_fusion_hidden_dim': 1,
        'interaction_hidden_dim': 1024,
        #'mlp_layers': [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
        #'final_mlp_layers': [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1],
        'num_factors': 8,
        'num_layers': 6,
        'dropout': 0.2
    }

    #_config['batch_size'] = 2048

    def train_vanilla():
        # Start pre-training
        print("======= Pre-training GMF =======")
        _config['gmf_checkpoint_path'] = _pretrain_gmf(_config)
        torch.cuda.empty_cache()

        print("======= Pre-training Vanilla MLP =======")
        _config['vanilla_mlp_checkpoint_path'] = _pretrain_vanilla_mlp(_config)
        torch.cuda.empty_cache()

        # Start training NeuMF
        print("======= Training NeuMF =======")
        neumf_best_path = _pretrain_neumf(_config)
        torch.cuda.empty_cache()

        print("======= Generating predictions =======")
        print("Predicting GMF")
        generate_predictions(_config['gmf_checkpoint_path'], _config, "/root/data")

        print("Predicting Vanilla MLP")
        generate_predictions(_config['vanilla_mlp_checkpoint_path'], _config, "/root/data")

        print("Predicting NeuMF")
        generate_predictions(neumf_best_path, _config, "/root/data")

        print("======= Getting metrics =======")
        print("GMF Metrics")
        get_metrics(pred_df_fpath="/root/data/gmf_predictions.parquet", true_df_fpath="/root/data/test.parquet")

        print("Vanilla MLP Metrics")
        get_metrics(pred_df_fpath="/root/data/vanilla_mlp_predictions.parquet",
                    true_df_fpath="/root/data/test.parquet")

        print("NeuMF Metrics")
        get_metrics(pred_df_fpath="/root/data/NeuMF_predictions.parquet",
                    true_df_fpath="/root/data/test.parquet")


    train_vanilla()






