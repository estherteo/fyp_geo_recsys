import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam, SGD


class NeuMF(pl.LightningModule):
    def __init__(self, num_users, num_items, num_factors, num_layers, dropout,
                 rating_threshold, GMF_model=None, MLP_model=None):
        """
        NeuMF combines the GMF and MLP models. Optionally, you can initialize it with
        pretrained GMF_model and MLP_model.
        """
        super(NeuMF, self).__init__()
        self.rating_threshold = rating_threshold
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        # GMF embeddings
        self.user_embedding_GMF = nn.Embedding(num_users, num_factors)
        self.item_embedding_GMF = nn.Embedding(num_items, num_factors)

        # MLP embeddings (same as in the MLP class)
        mlp_embedding_size = num_factors * (2 ** (num_layers - 1))
        self.user_embedding_MLP = nn.Embedding(num_users, mlp_embedding_size)
        self.item_embedding_MLP = nn.Embedding(num_items, mlp_embedding_size)

        # MLP layers
        modules = []
        for i in range(num_layers):
            input_size = num_factors * (2 ** (num_layers - i))
            modules.append(nn.Dropout(p=dropout))
            modules.append(nn.Linear(input_size, input_size // 2))
            modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*modules)
        # The prediction layer input is the concatenation of GMF and MLP outputs.
        self.predict_layer = nn.Linear(num_factors * 2, 1)
        self._init_weights()

        # Freeze embeddings if pretrained models are provided.
        if GMF_model is not None:
            for param in self.user_embedding_GMF.parameters():
                param.requires_grad = False
            for param in self.item_embedding_GMF.parameters():
                param.requires_grad = False

        if MLP_model is not None:
            for param in self.user_embedding_MLP.parameters():
                param.requires_grad = False
            for param in self.item_embedding_MLP.parameters():
                param.requires_grad = False

    def _init_weights(self):
        # If pretrained models are provided, load their weights.
        if self.GMF_model is not None and self.MLP_model is not None:
            # Copy embedding weights from pretrained models.
            self.user_embedding_GMF.weight.data.copy_(self.GMF_model.user_embedding.weight)
            self.item_embedding_GMF.weight.data.copy_(self.GMF_model.item_embedding.weight)
            self.user_embedding_MLP.weight.data.copy_(self.MLP_model.user_embedding.weight)
            self.item_embedding_MLP.weight.data.copy_(self.MLP_model.item_embedding.weight)
            # Copy MLP layers weights.
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            # Combine the pretrained prediction layers.
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight
            ], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)
        else:
            # Initialize from scratch.
            nn.init.xavier_uniform_(self.user_embedding_GMF.weight)
            nn.init.xavier_uniform_(self.item_embedding_GMF.weight)
            nn.init.normal_(self.user_embedding_MLP.weight, std=0.01)
            nn.init.normal_(self.item_embedding_MLP.weight, std=0.01)
            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
            if self.predict_layer.bias is not None:
                self.predict_layer.bias.data.zero_()

    def forward(self, user, item):
        # GMF branch
        user_embedding_GMF = self.user_embedding_GMF(user)
        item_embedding_GMF = self.item_embedding_GMF(item)
        output_GMF = user_embedding_GMF * item_embedding_GMF

        # MLP branch
        user_embedding_MLP = self.user_embedding_MLP(user)
        item_embedding_MLP = self.item_embedding_MLP(item)
        interaction = torch.cat((user_embedding_MLP, item_embedding_MLP), dim=-1)
        output_MLP = self.MLP_layers(interaction)

        # Concatenate GMF and MLP outputs and predict.
        concat = torch.cat((output_GMF, output_MLP), dim=-1)
        prediction = self.predict_layer(concat)
        return prediction

    def training_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('NeuMF_train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["review_stars"]
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            loss = F.binary_cross_entropy_with_logits(y_pred, labels.view(-1, 1))
        else:
            loss = F.mse_loss(y_pred, labels.view(-1, 1))
        self.log('NeuMF_val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        y_pred = self(batch['user_id'], batch['business_id'])
        if self.rating_threshold is not None:
            rating = torch.sigmoid(y_pred).flatten()  # shape: (batch_size,)
        else:
            rating = y_pred.flatten()
        return torch.stack((batch["user_id"], batch["business_id"], rating), dim=-1)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.01)

    def on_after_backward(self):
        # Log gradient norms for the embedding layers.
        for name, param in self.named_parameters():
            if ("embedding" in name) and param.grad is not None:
                self.log(f"neumf_{name}_norm", torch.norm(param.grad).item(), on_step=True, on_epoch=True)

