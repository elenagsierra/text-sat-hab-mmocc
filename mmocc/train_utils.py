from typing import cast

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import resize

from mmocc.config import weights_path
from mmocc.utils import de_normalize_transform


class SpeciesNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor

        speciesnet = torch.load(
            str(
                weights_path
                / "speciesnet_4.0.1b"
                / "full_image_88545560_22x8_v12_epoch_00153.pt"
            ),
            weights_only=False,
        )
        self.node = "SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__3839"
        self.speciesnet_feature_extractor = create_feature_extractor(
            speciesnet, [self.node]
        )

    def forward(self, x):

        # de-normalize the input
        x = de_normalize_transform(x)

        # convert from CHW to HWC format
        x = x.permute((0, 2, 3, 1))

        return self.speciesnet_feature_extractor(x)[self.node]


class MegaDetectorFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from PytorchWildlife.models import detection as pw_detection

        self.model = pw_detection.MegaDetectorV5().model
        del self.model.model[-1]  # remove the detection head

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 224
        x = resize(x, [256, 256])
        x = self.model.forward(x)
        x = torch.mean(x, dim=(2, 3))
        return x


class BioclipFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip"
        )

    def forward(self, x):
        return self.model.encode_image(x)  # type: ignore


class ClipFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: str):
        super().__init__()
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype
        self.device = next(self.model.parameters()).device
        self.clip_normalize = v2.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def forward(self, x):
        x = de_normalize_transform(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.clip_normalize(x)
        x = x.to(device=self.device, dtype=self.dtype)
        return self.model.encode_image(x)  # type: ignore


class FullyRandomFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return torch.randn(x.shape[0], self.output_dim, dtype=x.dtype, device=x.device)


def get_backbone(name: str) -> torch.nn.Module:
    if name == "dinov2_vitb14":
        return cast(
            torch.nn.Module, torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        )
    elif name == "dinov2_vits14":
        return cast(
            torch.nn.Module, torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        )
    elif name == "dinov3_vitb16":
        return cast(
            torch.nn.Module,
            torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vitb16",
                weights=str(
                    weights_path
                    / "dinov3"
                    / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
                ),
            ),
        )
    elif name == "dinov3_vitl16":
        return cast(
            torch.nn.Module,
            torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vitl16",
                weights=str(
                    weights_path
                    / "dinov3"
                    / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
                ),
            ),
        )
    elif name == "speciesnet":
        return SpeciesNetFeatureExtractor()
    elif name == "megadetector":
        return MegaDetectorFeatureExtractor()
    elif name == "bioclip":
        return BioclipFeatureExtractor()
    elif name == "clip_vitbigg14":
        return ClipFeatureExtractor("ViT-bigG-14", "laion2b_s39b_b160k")
    elif name == "random_resnet50":
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)

        # remove the classification head
        backbone.fc = torch.nn.Identity()  # type: ignore

        return backbone
    elif name == "fully_random":
        return FullyRandomFeatureExtractor()
    else:
        raise ValueError(f"Unknown backbone: {name}")


img_model_labels = {
    "bioclip": "BioCLIP",
    "megadetector": "MegaDetector",
    "speciesnet": "SpeciesNet",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "dinov3_vitl16": "DINOv3 ViT-L/16",
    "random_resnet50": "Random ResNet-50",
    "fully_random": "Noise",
    "clip_vitbigg14": "CLIP ViT-bigG/14 (LAION-2B)",
}

sat_model_labels = {
    "satbird": "SatBird",
    "prithvi_v2": "Prithvi-EO-2.0",
    "galileo": "Galileo",
    "tessera": "TESSERA",
    "alphaearth": "AlphaEarth Foundations",
    "taxabind": "TaxaBind",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    "dinov3_vitl16_sat": "DINOv3 ViT-L/16 (SAT-493M)",
    "random_resnet50": "Random ResNet-50",
    "fully_random": "Noise",
}
