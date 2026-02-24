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
"""Scrape expert habitat descriptors from the California Wildlife Habitat Relationships
System (CWHR) and save per-species habitat phrases for downstream analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fire
import pandas as pd
import requests

from mmocc.config import cache_path
from mmocc.utils import get_focal_species_ids, get_submitit_executor, get_taxon_map

SPECIES_ENDPOINT = "https://apps.wildlife.ca.gov/cwhr/handler.ashx"
ELEMENTS_ENDPOINT = SPECIES_ENDPOINT

PRIORITY_RANK = {"E": 0, "S": 1, "P": 2, "": 3}
PRIORITY_SCORE = {"E": 3.0, "S": 2.0, "P": 1.0, "": 0.0}
PRIORITY_LABEL = {
    "E": "Essential (E)",
    "S": "Secondarily Essential (S)",
    "P": "Preferred (P)",
    "": "Unspecified priority",
}

ANIMAL_KEYWORDS = {
    "animal",
    "amphib",
    "arthropod",
    "aves",
    "bird",
    "canid",
    "cervid",
    "crustacean",
    "egg",
    "fish",
    "insect",
    "invertebrate",
    "larva",
    "mammal",
    "pronghorn",
    "reptile",
    "rodent",
    "ungulate",
    "vertebrate",
    "carrion",
}

DEFAULT_OUTPUT_FILE = cache_path / "expert_habitat_descriptions.csv"
SPECIES_MAP_PATH = cache_path / "wi_db_computer_vision_species_map.csv"


def normalize_species_name(name: str) -> str:
    name = name.strip().lower()
    # Align spelling differences between data sources.
    name = name.replace("grey", "gray")
    return " ".join(name.split())


@dataclass
class HabitatElement:
    name: str
    definition: str
    priority: str

    @property
    def score(self) -> float:
        return PRIORITY_SCORE.get(self.priority, 0.0)

    @property
    def description(self) -> str:
        definition = self.definition.strip().replace(".", "")
        name_clean = self.name.lower().replace("layer, ", "").title()
        if definition:
            return f"{name_clean} ({definition})"
        return f"{name_clean}"


def fetch_species_id(session: requests.Session, species_name: str) -> str | None:
    params = {
        "cmd": "species_fillSpeciesList",
        "oby": "TAXA_SORT",
        "nf": "NAME",
        "whr": "NAME",
        "lim": species_name,
    }
    response = session.get(SPECIES_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        print(f"[WARN] Unexpected response for species lookup '{species_name}'.")
        return None
    if not isinstance(data, dict):
        print(
            f"[WARN] Unexpected payload type ({type(data)}) for species '{species_name}'."
        )
        return None
    items = data.get("items", [])
    if not items:
        return None

    normalized_target = normalize_species_name(species_name)
    for item in items:
        if normalize_species_name(item.get("NAME", "")) == normalized_target:
            return item.get("ID")

    print(f"[INFO] CWHR match not found for '{species_name}'. Skipping.")
    return None


def determine_priority(values: Sequence[str]) -> str:
    best_priority = ""
    best_rank = len(PRIORITY_RANK)
    for value in values:
        key = value.strip().upper()
        rank = PRIORITY_RANK.get(key, len(PRIORITY_RANK))
        if rank < best_rank:
            best_priority = key
            best_rank = rank
    return best_priority


def is_animal_element(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in ANIMAL_KEYWORDS)


def fetch_habitat_elements(
    session: requests.Session, cwhr_id: str
) -> List[HabitatElement]:
    params = {"cmd": "species_getElementsById", "cwhrid": cwhr_id, "pri": "A"}
    response = session.get(ELEMENTS_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError:
        print(f"[WARN] Unexpected habitat response for ID '{cwhr_id}'.")
        return []
    if not isinstance(data, dict):
        print(
            f"[WARN] Unexpected habitat payload type ({type(data)}) for ID '{cwhr_id}'."
        )
        return []
    items = data.get("items", [])
    elements: List[HabitatElement] = []
    for item in items:
        elem_name = item.get("elem_name", "Unknown Element")
        if is_animal_element(elem_name):
            continue
        priority = determine_priority(
            [item.get("repro", ""), item.get("cover", ""), item.get("feeding", "")]
        )
        elements.append(
            HabitatElement(
                name=elem_name,
                definition=item.get("definition", "").strip(),
                priority=priority,
            )
        )
    elements.sort(key=lambda elem: (PRIORITY_RANK.get(elem.priority, 3), elem.name))
    return elements


def load_species_metadata() -> pd.DataFrame:
    if not SPECIES_MAP_PATH.exists():
        raise FileNotFoundError(f"Species map not found at {SPECIES_MAP_PATH}")
    df = pd.read_csv(SPECIES_MAP_PATH)
    return df.set_index("WI_taxon_id")


def resolve_species_queries(taxon_ids: Sequence[str]) -> List[Tuple[str, str]]:
    taxon_map = get_taxon_map()
    metadata_df = load_species_metadata()

    queries: List[Tuple[str, str]] = []
    for taxon_id in taxon_ids:
        name = taxon_map.get(taxon_id)
        if not name:
            if taxon_id in metadata_df.index:
                candidate = metadata_df.at[taxon_id, "Scientific_Name"]
                name = (
                    candidate
                    if isinstance(candidate, str) and candidate.strip()
                    else None
                )
        if not name:
            print(
                f"[WARN] No taxonomy entry for taxon {taxon_id}; falling back to identifier."
            )
            name = taxon_id
        queries.append((taxon_id, name))
    return queries


def build_rows(
    session: requests.Session, species_entries: Iterable[Tuple[str, str]]
) -> List[Dict[str, str | float]]:
    rows: List[Dict[str, str | float]] = []
    missing_species: List[str] = []
    for taxon_id, species in species_entries:
        species_id = fetch_species_id(session, species)
        if not species_id:
            missing_species.append(f"{species} ({taxon_id})")
            continue

        try:
            elements = fetch_habitat_elements(session, species_id)
        except requests.HTTPError as exc:
            print(
                f"[WARN] Failed to fetch elements for {species} ({species_id}): {exc}"
            )
            continue

        if not elements:
            missing_species.append(species)
            continue

        for element in elements:
            rows.append(
                {
                    "taxon_id": taxon_id,
                    "species": species,
                    "difference": element.description,
                    "score": element.score,
                }
            )

    if missing_species:
        print(
            f"[INFO] No habitat elements found for {len(missing_species)} species: "
            + ", ".join(sorted(missing_species))
        )
    return rows


def retrieve_expert_descriptions(
    species_ids: Sequence[str] | None = None,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> Path:
    output_path = Path(output_file)
    if species_ids is None:
        species_ids = get_focal_species_ids()
    species_ids = list(species_ids)
    if not species_ids:
        raise RuntimeError("No species IDs provided.")

    species_entries = resolve_species_queries(species_ids)

    with requests.Session() as session:
        rows = build_rows(session, species_entries)

    if not rows:
        raise RuntimeError("No habitat elements were scraped; aborting.")

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} habitat element rows to {output_path}")
    return output_path


def main(
    species_ids: Sequence[str] | None = None,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
):
    executor = get_submitit_executor("retrieve_expert_descriptions")
    executor.update_parameters(
        slurm_mem="32G",
        cpus_per_task=4,
        slurm_additional_parameters=dict(gpus=0),
    )
    job = executor.submit(retrieve_expert_descriptions, species_ids, output_file)
    print(job.result())


if __name__ == "__main__":
    fire.Fire(main)
