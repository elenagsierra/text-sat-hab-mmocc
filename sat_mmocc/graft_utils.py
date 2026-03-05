"""Utilities for loading GRAFT models and CLIP-aligned text encoders.

Assumptions:
    - The GRAFT Sentinel checkpoint is trained to align with CLIP ViT-B/16.
    - Inputs are RGB images in [0, 1] and normalized with CLIP statistics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from torchvision import transforms

from mmocc.config import weights_path

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
GRAFT_EMBED_DIM = 512
GRAFT_DEFAULT_IMAGE_SIZE = 224
GRAFT_CHECKPOINT = weights_path / "graft" / "graft_sentinel.ckpt"


def _resolve_checkpoint(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[1] / "weights" / "graft" / "graft_sentinel.ckpt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"GRAFT checkpoint not found at {path} or {fallback}")


@dataclass(frozen=True)
class GraftConfig:
    """Configuration for loading GRAFT models."""

    checkpoint_path: Path = GRAFT_CHECKPOINT
    clip_model_name: str = CLIP_MODEL_NAME
    image_size: int = GRAFT_DEFAULT_IMAGE_SIZE


class GraftModel(nn.Module):
    """GRAFT satellite encoder built on CLIP ViT-B/16."""

    def __init__(self, clip_model_name: str = CLIP_MODEL_NAME):
        super().__init__()
        self.satellite_image_backbone = CLIPVisionModelWithProjection.from_pretrained(
            clip_model_name
        )
        hidden_size = self.satellite_image_backbone.config.hidden_size
        proj_dim = self.satellite_image_backbone.config.projection_dim
        eps = self.satellite_image_backbone.config.layer_norm_eps
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=eps),
            nn.Linear(hidden_size, proj_dim, bias=False),
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        hidden_state = self.satellite_image_backbone(image_tensor).last_hidden_state
        projected = self.projector(hidden_state)
        return F.normalize(projected, dim=-1)

    def forward_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        embed = self.satellite_image_backbone(image_tensor).image_embeds
        return F.normalize(embed, dim=-1)


def build_graft_transform(image_size: int = GRAFT_DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Build the CLIP-normalized transform expected by GRAFT."""

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def load_graft_model(
    config: GraftConfig | None = None, device: torch.device | None = None
) -> GraftModel:
    """Load the GRAFT model and checkpoint weights."""

    cfg = config or GraftConfig()
    model = GraftModel(cfg.clip_model_name)
    checkpoint_path = _resolve_checkpoint(cfg.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class GraftTextEncoder:
    """CLIP text encoder aligned with GRAFT embeddings."""

    def __init__(self, clip_model_name: str = CLIP_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
        for param in self.model.parameters():
            param.requires_grad_(False)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                tokens = self.tokenizer(
                    list(chunk), padding=True, return_tensors="pt"
                ).to(self.device)
                embeds = self.model(**tokens).text_embeds
                embeds = F.normalize(embeds, dim=-1)
                outputs.append(embeds.detach().cpu().float())
        return torch.cat(outputs, dim=0).numpy()


_TEXT_ENCODER: GraftTextEncoder | None = None


def get_graft_text_encoder() -> GraftTextEncoder:
    """Return a cached text encoder instance."""

    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _TEXT_ENCODER = GraftTextEncoder()
    return _TEXT_ENCODER
