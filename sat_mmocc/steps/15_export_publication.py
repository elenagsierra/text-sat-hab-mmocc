#!/usr/bin/env -S uv run --python 3.13 --script
#
# /// script
# dependencies = [
#     "mmocc",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# ///
"""Export codebase and test-only cache artifacts for publication."""

import fnmatch
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import fire
import numpy as np
import pandas as pd

from mmocc.config import (
    cache_path,
    default_image_backbone,
    default_sat_backbone,
    wildlife_insights_test_project_ids,
)
from mmocc.utils import filename_to_experiment

LOGGER = logging.getLogger(__name__)


def load_gitignore_patterns(repo_root: Path) -> list[str]:
    patterns: list[str] = []
    gitignore_path = repo_root / ".gitignore"
    if gitignore_path.exists():
        for line in gitignore_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            patterns.append(stripped)
    patterns.append(".export/")
    return patterns


def is_ignored(rel_path: Path, patterns: list[str]) -> bool:
    rel_posix = rel_path.as_posix()
    for pattern in patterns:
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            if any(fnmatch.fnmatch(part, prefix) for part in rel_path.parts):
                return True
            continue
        if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(
            rel_path.name, pattern
        ):
            return True
    return False


def iter_repo_files(repo_root: Path, patterns: list[str]) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(repo_root)
        if ".git" in rel.parts or ".export" in rel.parts:
            continue
        if is_ignored(rel, patterns):
            continue
        yield path


def list_git_files(repo_root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        LOGGER.warning("git ls-files failed, falling back to filesystem walk: %s", exc)
        return None
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        files.append(repo_root / raw.decode())
    return files


def copy_repo_files(repo_root: Path, export_root: Path, patterns: list[str]) -> int:
    files = list_git_files(repo_root)
    if files is None:
        files = list(iter_repo_files(repo_root, patterns))
    for path in files:
        destination = export_root / path.relative_to(repo_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return len(files)


def project_ids_from_loc_ids(ids: np.ndarray) -> np.ndarray:
    ids_str = ids.astype(str)
    return np.array([loc_id.split("___")[0] for loc_id in ids_str])


def build_test_mask(ids: np.ndarray) -> np.ndarray:
    project_ids = project_ids_from_loc_ids(ids)
    return np.isin(project_ids, list(wildlife_insights_test_project_ids))


def export_image_features(
    features_dir: Path, export_features_dir: Path, backbone: str
) -> int:
    prefix = f"wi_blank_image_features_{backbone}"
    ids_path = features_dir / f"{prefix}_ids.npy"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Missing image feature ids for backbone '{backbone}': {ids_path}"
        )
    ids = np.load(ids_path, allow_pickle=True)
    mask = build_test_mask(ids)
    if not mask.any():
        raise ValueError(f"No test entries found for backbone '{backbone}'.")

    sources = {
        f"{prefix}.npy": features_dir / f"{prefix}.npy",
        f"{prefix}_ids.npy": None,
        f"{prefix}_locs.npy": features_dir / f"{prefix}_locs.npy",
        f"{prefix}_covariates.npy": features_dir / f"{prefix}_covariates.npy",
    }

    for name, src in sources.items():
        if src is None:
            data = ids
        else:
            if not src.exists():
                raise FileNotFoundError(f"Missing required feature file: {src}")
            data = np.load(src, allow_pickle=True, mmap_mode="r")
        if data.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Feature length mismatch for {name}: {data.shape[0]} vs {mask.shape[0]}"
            )
        np.save(export_features_dir / name, data[mask])

    return int(mask.sum())


def export_sat_features(
    features_dir: Path, export_features_dir: Path, backbone: str
) -> int:
    prefix = f"wi_blank_sat_features_{backbone}"
    ids_path = features_dir / f"{prefix}_ids.npy"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Missing sat feature ids for backbone '{backbone}': {ids_path}"
        )
    ids = np.load(ids_path, allow_pickle=True)
    mask = build_test_mask(ids)
    if not mask.any():
        raise ValueError(f"No test entries found for sat backbone '{backbone}'.")

    sources = {
        f"{prefix}.npy": features_dir / f"{prefix}.npy",
        f"{prefix}_ids.npy": None,
    }

    for name, src in sources.items():
        if src is None:
            data = ids
        else:
            if not src.exists():
                raise FileNotFoundError(f"Missing required feature file: {src}")
            data = np.load(src, allow_pickle=True, mmap_mode="r")
        if data.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Feature length mismatch for {name}: {data.shape[0]} vs {mask.shape[0]}"
            )
        np.save(export_features_dir / name, data[mask])

    return int(mask.sum())


def export_fit_results(fit_results: Iterable[Path], export_cache_dir: Path) -> int:
    export_fit_dir = export_cache_dir / "fit_results"
    export_fit_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in fit_results:
        shutil.copy2(path, export_fit_dir / path.name)
        count += 1
    return count


def collect_backbones(fit_results: Iterable[Path]) -> tuple[set[str], set[str]]:
    image_backbones: set[str] = set()
    sat_backbones: set[str] = set()
    for path in fit_results:
        try:
            _, modalities, image_bb, sat_bb = filename_to_experiment(path.name)
        except ValueError:
            LOGGER.warning("Skipping unexpected fit results filename: %s", path.name)
            continue
        if "image" in modalities or "covariates" in modalities:
            backbone = (
                default_image_backbone if image_bb in {"None", "none", ""} else image_bb
            )
            if not "expert" in backbone and not "visdiff" in backbone:
                image_backbones.add(backbone)
        if "sat" in modalities:
            backbone = (
                default_sat_backbone if sat_bb in {"None", "none", ""} else sat_bb
            )
            sat_backbones.add(backbone)
    return image_backbones, sat_backbones


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parts = [item.strip() for item in value.split(",") if item.strip()]
    return parts if parts else None


def filter_fit_results(
    fit_results_dir: Path,
    image_backbones: list[str] | None,
    sat_backbones: list[str] | None,
    taxon_ids: list[str] | None,
) -> list[Path]:
    if not fit_results_dir.exists():
        LOGGER.warning("Fit results directory not found: %s", fit_results_dir)
        return []
    selected: list[Path] = []
    for path in fit_results_dir.glob("*.pkl"):
        try:
            taxon_id, modalities, image_bb, sat_bb = filename_to_experiment(path.name)
        except ValueError:
            LOGGER.warning("Skipping unexpected fit results filename: %s", path.name)
            continue
        if taxon_ids is not None and taxon_id not in taxon_ids:
            continue
        if "image" in modalities:
            image_name = (
                default_image_backbone if image_bb in {"None", "none", ""} else image_bb
            )
            if image_backbones is not None and image_name not in image_backbones:
                continue
        if "sat" in modalities:
            sat_name = (
                default_sat_backbone if sat_bb in {"None", "none", ""} else sat_bb
            )
            if sat_backbones is not None and sat_name not in sat_backbones:
                continue
        selected.append(path)
    return selected


def export_blank_images(export_cache_dir: Path) -> int:
    blank_images_path = cache_path / "wi_blank_images.pkl"
    if not blank_images_path.exists():
        LOGGER.warning("Missing blank image cache: %s", blank_images_path)
        return 0
    df = pd.read_pickle(blank_images_path)
    if "loc_id" not in df.columns:
        raise KeyError("wi_blank_images.pkl is missing the 'loc_id' column.")
    project_ids = df["loc_id"].astype(str).str.split("___").str[0]
    df_test = df.loc[project_ids.isin(wildlife_insights_test_project_ids)].copy()
    output_path = export_cache_dir / "wi_blank_images.pkl"
    df_test.to_pickle(output_path)
    return len(df_test)


def verify_exported_ids(export_features_dir: Path) -> None:
    test_project_ids = set(wildlife_insights_test_project_ids)
    violations = []
    for ids_path in export_features_dir.glob("*_ids.npy"):
        ids = np.load(ids_path, allow_pickle=True)
        project_ids = set(project_ids_from_loc_ids(ids))
        extra = project_ids - test_project_ids
        if extra:
            violations.append((ids_path.name, sorted(extra)[:5]))
    if violations:
        details = ", ".join(f"{name}: {extra}" for name, extra in violations)
        raise ValueError(f"Export contains non-test project ids: {details}")


def write_summary(export_cache_dir: Path, summary: dict) -> None:
    output_path = export_cache_dir / "export_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def main(
    output_dir: str = ".export",
    overwrite: bool = False,
    image_backbones: str | None = None,
    sat_backbones: str | None = None,
    taxon_ids: str | None = None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. Use --overwrite to replace."
            )

    output_path.mkdir(parents=True, exist_ok=True)
    patterns = load_gitignore_patterns(repo_root)
    num_files = copy_repo_files(repo_root, output_path, patterns)
    LOGGER.info("Copied %d repo files to %s", num_files, output_path)

    export_cache_dir = output_path / ".cache"
    export_cache_dir.mkdir(parents=True, exist_ok=True)

    fit_results_dir = cache_path / "fit_results"
    image_backbone_list = parse_csv_list(image_backbones)
    sat_backbone_list = parse_csv_list(sat_backbones)
    taxon_id_list = parse_csv_list(taxon_ids)
    selected_fit_results = filter_fit_results(
        fit_results_dir,
        image_backbone_list,
        sat_backbone_list,
        taxon_id_list,
    )
    num_fit_results = export_fit_results(selected_fit_results, export_cache_dir)
    image_backbones, sat_backbones = collect_backbones(selected_fit_results)

    features_dir = cache_path / "features"
    export_features_dir = export_cache_dir / "features"
    export_features_dir.mkdir(parents=True, exist_ok=True)

    feature_counts: dict[str, int] = {}
    for backbone in sorted(image_backbones):
        feature_counts[f"image_{backbone}"] = export_image_features(
            features_dir, export_features_dir, backbone
        )
    for backbone in sorted(sat_backbones):
        feature_counts[f"sat_{backbone}"] = export_sat_features(
            features_dir, export_features_dir, backbone
        )

    num_blank_images = export_blank_images(export_cache_dir)
    verify_exported_ids(export_features_dir)

    summary = dict(
        output=str(output_path),
        repo_files_copied=num_files,
        fit_results_copied=num_fit_results,
        blank_images_exported=num_blank_images,
        image_backbones=sorted(image_backbones),
        sat_backbones=sorted(sat_backbones),
        taxon_ids=taxon_id_list,
        feature_counts=feature_counts,
        test_project_ids=sorted(wildlife_insights_test_project_ids),
    )
    write_summary(export_cache_dir, summary)
    LOGGER.info("Export complete at %s", output_path)


if __name__ == "__main__":
    fire.Fire(main)
