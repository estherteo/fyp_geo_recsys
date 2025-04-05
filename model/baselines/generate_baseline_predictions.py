import pandas as pd
from pytorch_lightning import Trainer
from tqdm import tqdm

from model.baselines.dataloader import YelpDataModule
from model.baselines.layers.joint_gmf_model import JointGMF
from model.baselines.layers.joint_vanilla_mlp_model import JointVanillaMLP
from model.baselines.models.NeuMF import NeuMF
import torch


torch.set_float32_matmul_precision('high')


def generate_predictions(model_fpath: str, config, preds_outpath: str):
    datamodule = YelpDataModule(batch_size=config['batch_size'],
                                train_ds_fpath=config['train_ds_fpath'],
                                val_ds_fpath=config['val_ds_fpath'],
                                test_ds_fpath=config['test_ds_fpath'],
                                dataset_metadata_fpath=config['dataset_metadata_fpath'],
                                model_type='MLPInt',
                                threshold=config['RatingThreshold'],
                                train=False)

    datamodule.setup('predict')

    if 'vanilla_mlp' in model_fpath:
        model = JointVanillaMLP.load_from_checkpoint(
            model_fpath,
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_factors=config['num_factors'],
            num_layers=config['num_layers'],
            rating_threshold=config['RatingThreshold'],
            dropout=config['dropout'],
        )

    elif "gmf" in model_fpath:
        model = JointGMF.load_from_checkpoint(
            model_fpath,
            num_users=config['num_users'],
            num_items=config['num_items'],
            num_factors=config['num_factors'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            rating_threshold=config['RatingThreshold'],
        )

    elif 'NeuMF' in model_fpath:
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

        model = NeuMF.load_from_checkpoint(
            model_fpath,
            num_users=config['num_users'],
            num_items=config['num_items'],
            rating_threshold=config['RatingThreshold'],
            num_factors=config['num_factors'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            GMF_model=gmf,
            MLP_model=mlp
      )

    #model.eval()

    trainer = Trainer()
    dataloader = datamodule.test_dataloader()
    print("Number of samples to predict on:", len(dataloader.dataset))
    predictions = trainer.predict(model, dataloader)

    # Convert prediction tensors to 3 lists: user_ids, item_ids and predictions
    user_ids = []
    item_ids = []
    preds = []

    for batch in tqdm(predictions):
        batch = batch.detach().cpu()
        user_ids.extend(batch[:, 0].tolist())
        item_ids.extend(batch[:, 1].tolist())
        preds.extend(batch[:, 2].tolist())

    print("Number of predictions:", len(preds))

    df = pd.DataFrame(
        {
            'user_id': user_ids,
            'business_id': item_ids,
            'prediction': preds
        }
    )

    outpath = f"{preds_outpath}/{model_fpath.split('/')[-1].split('-')[0]}_predictions.parquet"

    print(f"Saving predictions to {outpath}")

    df.to_parquet(outpath)