#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "fire",
#     "numpy",
#     "pandas",
#     "tqdm",
#     "transformers",
#     "setuptools<81"
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Score GLC_FCS30D class-name text against pixel-level GRAFT token embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from mmocc.graft_utils import get_graft_text_encoder
from sat_mmocc.experiments.glc_fcs30d_graft.utils import (
    DEFAULT_GLC_YEAR,
    DEFAULT_IMAGERY_SOURCE,
    GLC_FILL_VALUE,
    build_label_texts,
    class_id_to_name,
    get_glc_class_table,
    get_run_dir,
)


def score_lulc_text(
    imagery_source: str = DEFAULT_IMAGERY_SOURCE,
    year: int = DEFAULT_GLC_YEAR,
    checkpoint_level: str = "pixel",
    top_k: int = 3,
    max_sites: int | None = None,
    skip_existing: bool = True,
) -> Path:
    run_dir = get_run_dir(
        imagery_source=imagery_source,
        year=year,
        checkpoint_level=checkpoint_level,
    )
    label_manifest_path = run_dir / "glc_label_manifest.csv"
    feature_manifest_path = run_dir / "pixel_feature_manifest.csv"
    if not label_manifest_path.exists():
        raise FileNotFoundError(
            f"Label manifest not found at {label_manifest_path}. "
            "Run step_01_download_glc_fcs30d_labels.py first."
        )
    if not feature_manifest_path.exists():
        raise FileNotFoundError(
            f"Feature manifest not found at {feature_manifest_path}. "
            "Run step_02_extract_pixel_graft_features.py first."
        )

    label_df = pd.read_csv(label_manifest_path)
    feature_df = pd.read_csv(feature_manifest_path)
    manifest_df = label_df.merge(
        feature_df,
        on="loc_id",
        how="inner",
        validate="one_to_one",
    )
    if max_sites is not None:
        manifest_df = manifest_df.head(max_sites).copy()

    outputs_dir = run_dir / "text_scoring"
    predictions_dir = outputs_dir / "site_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    class_table = get_glc_class_table(include_fill=False)
    class_ids = class_table["class_id"].to_numpy(dtype=np.int16)
    class_names = class_table["class_name"].tolist()
    class_id_to_index = {int(class_id): idx for idx, class_id in enumerate(class_ids)}

    text_labels = build_label_texts(include_fill=False)
    text_embeddings = get_graft_text_encoder().encode(text_labels).astype(
        np.float32,
        copy=False,
    )

    np.save(outputs_dir / "class_name_embeddings.npy", text_embeddings)
    class_table.assign(text_label=text_labels).to_csv(
        outputs_dir / "class_name_table.csv",
        index=False,
    )

    confusion = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    site_rows: list[dict[str, object]] = []
    total_tokens = 0
    total_correct = 0
    total_topk_correct = 0
    total_skipped = 0

    for _, row in tqdm(
        manifest_df.iterrows(),
        total=len(manifest_df),
        desc="Scoring text labels",
    ):
        loc_id = str(row["loc_id"])
        prediction_path = predictions_dir / f"{loc_id}.npz"
        if skip_existing and prediction_path.exists():
            cached = np.load(prediction_path, allow_pickle=False)
            true_class_ids = cached["true_class_ids"]
            predicted_class_ids = cached["predicted_class_ids"]
            topk_hit = cached["topk_hit"]
            confidences = cached["confidence"]
        else:
            feature_grid = np.load(Path(row["feature_path"]), allow_pickle=False).astype(
                np.float32,
                copy=False,
            )
            true_class_ids = np.load(
                Path(row["token_label_path"]),
                allow_pickle=False,
            ).astype(np.int16, copy=False)
            if feature_grid.shape[:2] != true_class_ids.shape:
                raise ValueError(
                    f"Feature/label shape mismatch for {loc_id}: "
                    f"{feature_grid.shape[:2]} vs {true_class_ids.shape}"
                )

            valid_mask = true_class_ids != GLC_FILL_VALUE
            if not np.any(valid_mask):
                total_skipped += 1
                continue

            flat_features = feature_grid.reshape(-1, feature_grid.shape[-1])
            scores = flat_features @ text_embeddings.T
            predicted_indices = scores.argmax(axis=1)
            predicted_class_ids = class_ids[predicted_indices].reshape(true_class_ids.shape)
            confidences = scores.max(axis=1).reshape(true_class_ids.shape).astype(
                np.float32,
                copy=False,
            )
            if top_k > 1:
                topk_indices = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
                topk_class_ids = class_ids[topk_indices]
                topk_hit = np.any(
                    topk_class_ids == true_class_ids.reshape(-1, 1),
                    axis=1,
                ).reshape(true_class_ids.shape)
            else:
                topk_hit = predicted_class_ids == true_class_ids

            np.savez_compressed(
                prediction_path,
                true_class_ids=true_class_ids.astype(np.int16, copy=False),
                predicted_class_ids=predicted_class_ids.astype(np.int16, copy=False),
                topk_hit=topk_hit.astype(bool, copy=False),
                confidence=confidences.astype(np.float32, copy=False),
            )

        valid_mask = true_class_ids != GLC_FILL_VALUE
        valid_true = true_class_ids[valid_mask]
        valid_pred = predicted_class_ids[valid_mask]
        valid_topk_hit = np.asarray(topk_hit[valid_mask], dtype=bool)
        valid_confidence = np.asarray(confidences[valid_mask], dtype=np.float32)

        true_indices = np.array(
            [class_id_to_index[int(v)] for v in valid_true],
            dtype=np.int64,
        )
        pred_indices = np.array(
            [class_id_to_index[int(v)] for v in valid_pred],
            dtype=np.int64,
        )
        np.add.at(confusion, (true_indices, pred_indices), 1)

        num_tokens = int(valid_mask.sum())
        correct = int((valid_true == valid_pred).sum())
        topk_correct = int(valid_topk_hit.sum())
        total_tokens += num_tokens
        total_correct += correct
        total_topk_correct += topk_correct

        dominant_true = int(pd.Series(valid_true).mode().iloc[0])
        dominant_pred = int(pd.Series(valid_pred).mode().iloc[0])
        site_rows.append(
            {
                "loc_id": loc_id,
                "glc_year": int(row["glc_year"]),
                "num_valid_tokens": num_tokens,
                "token_accuracy": correct / num_tokens if num_tokens else np.nan,
                "topk_accuracy": topk_correct / num_tokens if num_tokens else np.nan,
                "mean_confidence": float(valid_confidence.mean()) if num_tokens else np.nan,
                "dominant_true_class_id": dominant_true,
                "dominant_true_class_name": class_id_to_name(dominant_true),
                "dominant_pred_class_id": dominant_pred,
                "dominant_pred_class_name": class_id_to_name(dominant_pred),
                "prediction_path": str(prediction_path),
            }
        )

    site_metrics_df = pd.DataFrame(site_rows).sort_values("loc_id").reset_index(drop=True)
    site_metrics_path = outputs_dir / "site_metrics.csv"
    site_metrics_df.to_csv(site_metrics_path, index=False)

    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    confusion_df.to_csv(outputs_dir / "confusion_matrix.csv")

    summary = {
        "imagery_source": imagery_source,
        "glc_year": int(year),
        "checkpoint_level": checkpoint_level,
        "text_prompt_mode": "raw_category_names",
        "num_sites_scored": int(len(site_metrics_df)),
        "num_sites_skipped_no_valid_labels": int(total_skipped),
        "num_tokens_scored": int(total_tokens),
        "token_accuracy": (total_correct / total_tokens) if total_tokens else None,
        "topk": int(top_k),
        "topk_accuracy": (total_topk_correct / total_tokens) if total_tokens else None,
    }
    with open(outputs_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote text-scoring outputs to {outputs_dir}")
    return outputs_dir


if __name__ == "__main__":
    fire.Fire(score_lulc_text)
