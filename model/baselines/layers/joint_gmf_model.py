import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class JointGMF(pl.LightningModule):
    def __init__(self, num_users, num_items, num_factors, rating_threshold):
        """
        GMF model uses element-wise product of user and item embeddings.
        """
        super(JointGMF, self).__init__()
        self.rating_threshold = rating_threshold

        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)
        self.predict_layer = nn.Linear(num_factors, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.predict_layer.weight)
        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        interaction = user_embed * item_embed
        prediction = self.predict_layer(interaction)
        return prediction

    def training_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('GMF_train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('GMF_val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            rating = torch.sigmoid(y_pred).flatten()  # shape: (batch_size,)
        else:
            rating = y_pred.flatten()
        return torch.stack((batch["user_id"], batch["business_id"], rating), dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def on_after_backward(self):
        # Log gradient norms for the embedding layers.
        for name, param in self.named_parameters():
            if ("embedding" in name) and param.grad is not None:
                self.log(f"{name}_norm", torch.norm(param.grad).item(), on_step=True, on_epoch=True)