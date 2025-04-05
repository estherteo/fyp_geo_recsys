import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class JointVanillaMLP(pl.LightningModule):
    def __init__(self, num_users, num_items, num_factors, num_layers, rating_threshold, dropout):
        """
        JointVanillaMLP model uses concatenated user and item embeddings followed by a multilayer perceptron.
        """
        super(JointVanillaMLP, self).__init__()

        self.rating_threshold = rating_threshold

        mlp_embedding_size = num_factors * (2 ** (num_layers - 1))
        self.user_embedding = nn.Embedding(num_users, mlp_embedding_size)
        self.item_embedding = nn.Embedding(num_items, mlp_embedding_size)

        modules = []
        for i in range(num_layers):
            input_size = num_factors * (2 ** (num_layers - i))
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(input_size, input_size // 2))
            modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*modules)
        # After the JointVanillaMLP layers the output size becomes num_factors.
        self.predict_layer = nn.Linear(num_factors, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        if self.predict_layer.bias is not None:
            self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        interaction = torch.cat((user_embed, item_embed), dim=-1)
        mlp_output = self.MLP_layers(interaction)
        prediction = self.predict_layer(mlp_output)
        return prediction

    def training_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('Vanilla_MLP_train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('Vanilla_MLP_val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            rating = torch.sigmoid(y_pred).flatten()  # shape: (batch_size,)
        else:
            rating = y_pred.flatten()
        return torch.stack((batch["user_id"], batch["business_id"], rating), dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)

    def on_after_backward(self):
        # Log gradient norms for the embedding layers.
        for name, param in self.named_parameters():
            if ("embedding" in name) and param.grad is not None:
                self.log(f"vanilla_mlp_{name}_norm", torch.norm(param.grad).item(), on_step=True, on_epoch=True)
