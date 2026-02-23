"""Create a UMAP or t-SNE plot of CLIP embeddings for top habitat descriptors."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from mmocc.config import (
    cache_path,
    fig_page_width,
    figures_path,
    num_habitat_descriptions,
)
from mmocc.plot_utils import setup_matplotlib
from mmocc.utils import get_focal_species_ids, get_scientific_taxon_map

CLIP_MODEL_NAME = "ViT-bigG-14"
CLIP_PRETRAINED = "laion2b_s39b_b160k"

DESCRIPTOR_PATHS: Dict[str, Path] = {
    "VisDiff": cache_path / "visdiff_sat_descriptions.csv",
    "Expert": cache_path / "expert_habitat_descriptions.csv",
}


@dataclass(frozen=True)
class DescriptorRow:
    taxon_id: str
    species_name: str
    source: str
    description: str
    score: float


class ClipTextEncoder:
    """Lazy CLIP text encoder used for descriptor embeddings."""

    def __init__(self, model_name: str, pretrained: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        self.model = model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.device = device

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        tokens = open_clip.tokenize(list(texts))
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = tokens[start : start + batch_size].to(self.device)
                features = self.model.encode_text(chunk)  # type: ignore[attr-defined]
                features = torch.nn.functional.normalize(features, dim=-1)
                outputs.append(features.detach().cpu().float())
        return torch.cat(outputs, dim=0).numpy()


_TEXT_ENCODER: ClipTextEncoder | None = None


def get_text_encoder() -> ClipTextEncoder:
    global _TEXT_ENCODER
    if _TEXT_ENCODER is None:
        _TEXT_ENCODER = ClipTextEncoder(CLIP_MODEL_NAME, CLIP_PRETRAINED)
    return _TEXT_ENCODER


def _clean_descriptors(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    df = df.copy()
    df["taxon_id"] = df["taxon_id"].astype(str)
    df["difference"] = df["difference"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df.get("score", 1.0), errors="coerce")
    df = df.dropna(subset=["taxon_id", "difference", "score"])
    df = df[df["difference"] != ""]
    df = df.sort_values(["taxon_id", "score"], ascending=[True, False])
    df = df.drop_duplicates(subset=["taxon_id", "difference"])
    df = df.groupby("taxon_id").head(limit)
    return df.reset_index(drop=True)


def load_descriptors(
    focal_ids: Sequence[str],
    taxon_map: Dict[str, str],
    descriptor_limit: int,
) -> list[DescriptorRow]:
    rows: list[DescriptorRow] = []
    focal_id_set = set(focal_ids)
    for source, path in DESCRIPTOR_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing descriptor cache at {path}")
        raw = pd.read_csv(path)
        cleaned = _clean_descriptors(raw, descriptor_limit)
        cleaned = cleaned[cleaned["taxon_id"].isin(focal_id_set)]
        for _, row in cleaned.iterrows():
            taxon_id = str(row["taxon_id"])
            name = taxon_map.get(taxon_id)
            if not name:
                name = taxon_id
            rows.append(
                DescriptorRow(
                    taxon_id=taxon_id,
                    species_name=name,
                    source=source,
                    description=str(row["difference"]),
                    score=float(row["score"]),
                )
            )
    if not rows:
        raise RuntimeError("No descriptors found for the selected focal species.")
    return rows


def build_color_map(species_names: Iterable[str]) -> Dict[str, tuple]:
    unique_names = sorted(set(species_names))
    palette = sns.color_palette("tab20", n_colors=len(unique_names))
    return dict(zip(unique_names, palette))


def plot_umap(
    coords: np.ndarray,
    descriptors: list[DescriptorRow],
    color_map: Dict[str, tuple],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(fig_page_width, fig_page_width))
    marker_map = {"VisDiff": "o", "Expert": "s"}

    df = pd.DataFrame([d.__dict__ for d in descriptors])
    df["color"] = df["species_name"].map(color_map)

    for source, marker in marker_map.items():
        subset = df[df["source"] == source]
        ax.scatter(
            coords[subset.index, 0],
            coords[subset.index, 1],
            label=source,
            c=subset["color"].tolist(),
            edgecolors="white",
            linewidths=0.6,
            alpha=0.9,
            s=55,
            marker=marker,
        )

    species_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[name],
            markeredgecolor="white",
            markersize=7,
            label=name,
        )
        for name in sorted(color_map.keys())
    ]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_map[src],
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=7,
            label=f"{src} description",
        )
        for src in marker_map
    ]

    species_legend = ax.legend(
        handles=species_handles,
        title="Species",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
    )
    ax.add_artist(species_legend)
    ax.legend(
        handles=source_handles,
        title="Descriptor source",
        bbox_to_anchor=(1.02, 0.55),
        loc="upper left",
        frameon=False,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Habitat descriptor UMAP (CLIP text space)")
    ax.grid(False)
    fig.tight_layout()
    return fig


def compute_umap_embeddings(texts: Sequence[str]) -> np.ndarray:
    if len(texts) < 2:
        raise ValueError("At least two descriptors are required for UMAP.")
    encoder = get_text_encoder()
    embeddings = encoder.encode(texts)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=0,
        n_neighbors=min(15, max(2, len(texts) - 1)),
        min_dist=0.1,
    )
    return reducer.fit_transform(embeddings)


def compute_tsne_embeddings(texts: Sequence[str]) -> np.ndarray:
    if len(texts) < 2:
        raise ValueError("At least two descriptors are required for t-SNE.")
    encoder = get_text_encoder()
    embeddings = encoder.encode(texts)
    # Perplexity must be < number of samples; keep it conservative for small sets.
    perplexity = max(5, min(30, len(texts) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=0,
        metric="cosine",
    )
    return tsne.fit_transform(embeddings)


def main(
    descriptor_limit: int = num_habitat_descriptions, method: str = "umap"
) -> None:
    setup_matplotlib()
    focal_ids = get_focal_species_ids()
    taxon_map = get_scientific_taxon_map()
    descriptors = load_descriptors(focal_ids, taxon_map, descriptor_limit)
    texts = [d.description for d in descriptors]
    method = method.lower()
    if method == "umap":
        coords = compute_umap_embeddings(texts)
    elif method == "tsne":
        coords = compute_tsne_embeddings(texts)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'umap' or 'tsne'.")
    color_map = build_color_map(d.species_name for d in descriptors)
    fig = plot_umap(coords, descriptors, color_map)
    output_png = figures_path / f"habitat_covariates_{method}.png"
    output_pdf = figures_path / f"habitat_covariates_{method}.pdf"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved {method.upper()} plot to {output_png} and {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project habitat descriptors with CLIP and UMAP/t-SNE."
    )
    parser.add_argument(
        "--descriptor-limit",
        type=int,
        default=num_habitat_descriptions,
        help="How many top descriptors per species to include.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "tsne"],
        help="Dimensionality reduction method.",
    )
    args = parser.parse_args()
    main(descriptor_limit=args.descriptor_limit, method=args.method)
