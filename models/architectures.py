import math
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as torch_models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GrayscaleToPseudoRGB(nn.Module):
    def __init__(self):
        super(GrayscaleToPseudoRGB, self).__init__()

    def forward(self, x):
        # x is grayscale with shape [N, 1, H, W]
        return x.repeat(1, 3, 1, 1)  # Output shape [N, 3, H, W]


def returnModel(pretrain, replicate):
    """

    This function returns the model with the final FF layers removed

    """
    model = torch_models.resnet50(pretrained=pretrain)
    model.fc = Identity()
    if not replicate:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        model.conv1 = nn.Sequential(
            GrayscaleToPseudoRGB(),
            model.conv1
        )
        # model.conv1[1].in_channels = 1
    return model


class Pvas_Model(nn.Module):
    def __init__(self, pretrain, replicate, dropout=0.5, split=True):
        super(Pvas_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate)
        self.D = 2048  ## out of the model
        self.K = 1024  ## intermidiate
        self.L = 512
        self.split = split
        if not split:
            self.D += 2

        ## Standard regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.K, self.L),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.L, 1)
        )

    ## Feed forward function
    def forward(self, x, is_it_mlo):
        H = self.extractor(x)
        if not self.split:
            H = torch.vstack([H.T, is_it_mlo]).T
        r = self.regressor(H)
        return r

    ## This is used for loss calculation and training, do not call!
    def objective(self, X, Y):
        Y = Y.unsqueeze(1)
        R = self.forward(X)
        loss = nn.MSELoss()
        return loss(R, Y), R

    ## This is a simple inference function for testing
    def apply(self, X):
        return self.forward(X)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 32 * 16 * 10, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(64, 1, bias=False)  # Regression output
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# define complex resnet into transformer model
class ResNetTransformer(nn.Module):
    def __init__(self, pretrain, replicate, dropout=0.5, split=True):
        super(ResNetTransformer, self).__init__()

        # Using ResNet-34 as a feature extractor
        self.resnet = resnet34(pretrained=pretrain)  # set pretrained to False
        d_model = 512
        nhead = 4  # Number of self-attention heads
        num_encoder_layers = 2  # Number of Transformer encoder layers
        self.D = d_model  ## out of the model
        self.K = 1024  ## intermidiate
        self.L = 512
        self.split = split
        if not split:
            self.D += 2

        # Modify the first layer to accept single-channel (grayscale) images
        # self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if not replicate:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.resnet.conv1 = nn.Sequential(
                GrayscaleToPseudoRGB(),
                self.resnet.conv1
            )
            # model.conv1[1].in_channels = 1

        self.resnet.fc = nn.Identity()  # Removing the fully connected layer

        # Assuming we are using an average pool and get a 512-dimensional vector

        # Transformer Encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Final regressor

        ## Standard regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.K, self.L),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.L, 1)
        )

    def forward(self, x, is_it_mlo):
        # Extract features using ResNet
        x = self.resnet(x)
        x = x.unsqueeze(1)  # Add sequence length dimension for Transformer

        # Pass features through Transformer
        x = self.transformer_encoder(x)

        # Regression
        x = x.squeeze(1)
        if not self.split:
            x = torch.vstack([x.T, is_it_mlo]).T
        x = self.regressor(x)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=24, embed_dim=512, bias=True):
        super().__init__()
        self.patch_size = patch_size
        # self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=(patch_size, patch_size),
                              stride=(int(patch_size/1), int(patch_size/1)),
                              padding=(patch_size, patch_size),
                              bias=False)
        if not bias:
            nn.init.constant_(self.proj.weight, 1 / (patch_size * patch_size))

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=640):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=8, num_classes=1, epsilon=0):
        super(TransformerModel, self).__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.mask_generator = PatchEmbedding(embed_dim=embed_dim, bias=False)
        self.epsilon = epsilon
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=True)

    def generate_padding_mask(self, x):
        # Assuming 0 is used for padding in patch embeddings
        # Calculate the mean across the embedding dimension, if it's 0, it's a padding patch
        mask = x.mean(dim=-1) > self.epsilon
        # print('total masked = ', torch.sum(mask.float() - 1))
        return mask

    def forward(self, x):
        # Generate padding mask
        empty_check = self.mask_generator(x)
        mask = self.generate_padding_mask(empty_check)

        x = self.patch_embed(x)  # Patch embedding
        x = self.pos_encoder(x)  # Add positional encoding

        # Transformer expects the mask to have a different shape, so we modify it accordingly
        # mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask.float()) * -10000.0  # Convert to attention scores

        x = self.transformer_encoder(x, src_key_padding_mask=mask.T)  # Transformer encoder
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)  # Classifier

        return x