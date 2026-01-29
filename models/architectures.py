import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet34
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torchvision.models as torch_models


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Args:
            alpha (float, tensor): Weighting factor for the class imbalance
            gamma (float): Focusing parameter
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (before softmax) of shape (N, C) where C is the number of classes
            targets: Ground-truth labels of shape (N,)

        Returns:
            Computed focal loss value
        """
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Compute probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction

class MILPooling(nn.Module):
    def __init__(self, feature_dim, pooling_type='mean'):
        super(MILPooling, self).__init__()
        self.pooling_type = pooling_type
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features):
        if self.pooling_type == 'mean':
            return features.mean(dim=0)
        elif self.pooling_type == 'max':
            return features.max(dim=0)[0]
        elif self.pooling_type == 'attention':
            return self.attention_pooling(features)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    def attention_pooling(self, features):
        # Implementing a simple attention mechanism
        attention_weights = self.attention_fc(features)
        attention_weights = torch.softmax(attention_weights, dim=0)
        weighted_features = features * attention_weights
        weighted_attention_matrix = weighted_features.sum(dim=0)
        return weighted_attention_matrix


class CRUK_MIL_Model(nn.Module):
    def __init__(self, pretrain, replicate, resnet_size, pooling_type='attention', dropout=0.5,
                 split=False, num_classes=15):
        super(CRUK_MIL_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate, resnet_size)
        if resnet_size in [18, 34]:
            self.D = 512
        elif resnet_size in [50, 101, 152]:
            self.D = 2048
        self.K = 1024  # intermediate
        self.L = 512
        self.MIL = MILPooling(self.L, pooling_type)
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
        )
        self.output = nn.Linear(self.L, num_classes*2)  # n binary classifications squished together

    ## Feed forward function
    def forward(self, image_data):
        image_features = []
        for image, view in image_data:
            image = image.to('cuda')
            is_it_mlo = torch.stack([torch.tensor([0, 1] if v == 'MLO' else [1, 0]).to('cuda') for v in view])
            H = self.extractor(image.unsqueeze(1))
            if not self.split:
                H = torch.hstack([H, is_it_mlo])
            r = self.regressor(H.to(torch.float32))
            image_features.append(r)
        image_features = torch.stack(image_features)
        mil = self.MIL(image_features)
        output = self.output(mil)
        return output


class Recurrence_MIL_Model(nn.Module):
    def __init__(self, pretrain, replicate, resnet_size, pooling_type='attention', dropout=0.5, split=True,
                 num_manufacturers=6, num_classes=4, include_vas=True):
        super(Recurrence_MIL_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate, resnet_size)
        if resnet_size in [18, 34]:
            self.D = 512
        elif resnet_size in [50, 101, 152]:
            self.D = 2048
        self.K = 1024  # intermediate
        self.L = 512
        self.MIL = MILPooling(self.L, pooling_type)
        self.split = split
        if not split:
            self.D += 2
        self.include_vas = include_vas
        if self.include_vas:
            self.D += 1
        self.D += 1  # for timepoint
        self.D += num_manufacturers

        ## Standard regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.K, self.L),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.output = nn.Linear(self.L, num_classes*2)  # n binary classifications squished together

    ## Feed forward function
    def forward(self, image_data, manufacturer_mapping):
        image_features = []
        for image, score, timepoint, patient, manu, view in image_data:
            image, score, timepoint = image.to('cuda'), score.to('cuda'), timepoint.to('cuda')  # Send data to GPU
            is_it_mlo = torch.stack([torch.tensor([0, 1] if v == 'mlo' else [1, 0]).to('cuda') for v in view])
            manu_key = torch.stack([manufacturer_mapping[m] for m in manu])
            H = self.extractor(image.unsqueeze(1))
            if not self.split:
                if self.include_vas:
                    H = torch.hstack([H, is_it_mlo, manu_key, timepoint.unsqueeze(1), score.unsqueeze(1)])
                else:
                    H = torch.hstack([H, is_it_mlo, manu_key, timepoint.unsqueeze(1)])
            r = self.regressor(H.to(torch.float32))
            image_features.append(r)
        image_features = torch.stack(image_features)
        mil = self.MIL(image_features)
        output = self.output(mil)
        return output


class Medici_MIL_Model(nn.Module):
    def __init__(self, pretrain, replicate, resnet_size, pooling_type='attention', dropout=0.5, split=True,
                 num_manufacturers=6):
        super(Medici_MIL_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate, resnet_size)
        if resnet_size in [18, 34]:
            self.D = 512
        elif resnet_size in [50, 101, 152]:
            self.D = 2048
        self.K = 1024  ## intermidiate
        self.L = 512
        self.MIL = MILPooling(self.L, pooling_type)
        self.split = split
        if not split:
            self.D += 2
        self.D += num_manufacturers

        ## Standard regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.K),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.K, self.L),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )
        self.output = nn.Linear(self.L, 1)

    ## Feed forward function
    def forward(self, xs, is_it_mlos, manus):
        image_features = []
        for i in range(is_it_mlos.shape[1]):
            x, is_it_mlo, manu = xs[:, :, i], is_it_mlos[:, i], manus[:, i]
            H = self.extractor(x)
            if not self.split:
                H = torch.hstack([H, is_it_mlo, manu])
            r = self.regressor(H)
            image_features.append(r)
        image_features = torch.stack(image_features)
        mil = self.MIL(image_features)
        output = self.output(mil)
        return output

class Mosaic_MIL_Model(nn.Module):
    def __init__(self, pretrain, replicate, resnet_size, pooling_type='attention', dropout=0.5, split=True):
        super(Mosaic_MIL_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate, resnet_size)
        if resnet_size in [18, 34]:
            self.D = 512
        elif resnet_size in [50, 101, 152]:
            self.D = 2048
        self.K = 1024  ## intermidiate
        self.L = 512
        self.MIL = MILPooling(self.L, pooling_type)
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
        )
        self.output = nn.Linear(self.L, 1)

    ## Feed forward function
    def forward(self, xs, is_it_mlos):
        image_features = []
        for i in range(is_it_mlos.shape[1]):
            x, is_it_mlo = xs[:, :, i], is_it_mlos[:, i]
            H = self.extractor(x)
            if not self.split:
                H = torch.hstack([H, is_it_mlo])
            r = self.regressor(H)
            image_features.append(r)
        image_features = torch.stack(image_features)
        mil = self.MIL(image_features)
        output = self.output(mil)
        return output


class Mosaic_PVAS_Model(nn.Module):
    def __init__(self, pretrain, replicate, resnet_size, pooling_type='attention', dropout=0.5, split=True):
        super(Mosaic_PVAS_Model, self).__init__()

        self.replicate = replicate
        self.extractor = returnModel(pretrain, replicate, resnet_size)
        if resnet_size in [18, 34]:
            self.D = 512
        elif resnet_size in [50, 101, 152]:
            self.D = 2048
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
    def forward(self, xs, is_it_mlos):
        predictions = 0
        for i in range(is_it_mlos.shape[1]):
            x, is_it_mlo = xs[:, :, i], is_it_mlos[:, i]
            H = self.extractor(x)
            if not self.split:
                H = torch.hstack([H, is_it_mlo])
            r = self.regressor(H)
            predictions += r
        output = predictions / is_it_mlos.shape[1]
        return output


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


def returnModel(pretrain, replicate, resnet_size=50):
    """

    This function returns the model with the final FF layers removed

    """
    if resnet_size == 18:
        model = models.resnet18(pretrained=pretrain)
    elif resnet_size == 34:
        model = models.resnet34(pretrained=pretrain)
    elif resnet_size == 50:
        model = models.resnet50(pretrained=pretrain)
    else:
        print("Not a valid resnet size")
        Exception
    model.fc = Identity()
    if not replicate:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # else:
    #     model.conv1 = nn.Sequential(
    #         GrayscaleToPseudoRGB(),
    #         model.conv1
    #     )
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
            H = torch.hstack([H, is_it_mlo])
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
            x = torch.hstack([x, is_it_mlo])
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