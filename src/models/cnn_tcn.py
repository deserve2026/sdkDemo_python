from torch import nn
from src.models.cnn_backbone import make_feature_extractor
from src.models.tcn import build_tcn


class CNNTCN(nn.Module):
    """Optional composition; baseline training uses cached features instead."""
    def __init__(self, backbone, tcn):
        super().__init__()
        self.backbone, self.tcn = backbone, tcn

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        import torch
        batch, time, channels, height, width = images.shape
        with torch.no_grad():
            features = self.backbone(images.reshape(batch * time, channels, height, width))
        return self.tcn(features.reshape(batch, time, -1))


def build_model(dataset_config, model_config):
    return CNNTCN(make_feature_extractor(dataset_config), build_tcn(model_config))

