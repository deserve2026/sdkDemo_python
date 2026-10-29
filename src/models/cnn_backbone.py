"""Prepared loader for the later experiment; never invoked by run_pipeline."""
from src.utils.paths import output_path


def make_feature_extractor(config, pretrained=True):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    # Keep automatic weight downloads under the requested results directory.
    cache_dir = output_path(config, 'checkpoints/pretrained/cache_anchor').parent
    torch.hub.set_dir(str(cache_dir))
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    return model

