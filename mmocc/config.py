import logging
import os
from collections import defaultdict
from multiprocessing import set_start_method
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
set_start_method("spawn", force=True)

load_dotenv(find_dotenv(usecwd=True))

cache_path = Path(os.environ["CACHE_PATH"])
log_path = Path(os.environ["LOG_PATH"])
wi_metadata_path = Path(os.environ["WI_METADATA_PATH"])
wi_image_path = Path(os.environ["WI_IMAGE_PATH"])
weights_path = Path(os.environ["WEIGHTS_PATH"])
figures_path = Path(os.environ["FIGURES_PATH"])

cache_path.mkdir(parents=True, exist_ok=True)
log_path.mkdir(parents=True, exist_ok=True)
wi_metadata_path.mkdir(parents=True, exist_ok=True)

image_batch_size = 16
sat_batch_size = 16

# Covariates used by SatBird (Teng et al. 2023)
# see: https://arxiv.org/pdf/2311.00936
satbird_covariates = [
    "bio01",
    "bio02",
    "bio03",
    "bio04",
    "bio05",
    "bio06",
    "bio07",
    "bio08",
    "bio09",
    "bio10",
    "bio11",
    "bio12",
    "bio13",
    "bio14",
    "bio15",
    "bio16",
    "bio17",
    "bio18",
    "bio19",
    "orcdrc",
    "phihox",
    "cecsol",
    # "bdticm",  # not available on GEE
    "clyppt",
    "sltppt",
    "sndppt",
    "bldfie",
]

default_covariates = satbird_covariates

# Wildlife Insights Snapshot USA project IDs for testing
wildlife_insights_test_project_ids = {
    "145625598251_2004778_840_p287",
    "437283855702_2003286_4435_snapshot_usa_2021",
    "145625598251_2004415_858_snapshot_usa_2022",
    "145625598251_2006496_506_snapshot_usa_2023",
}

# Wildlife Insights project IDs to exclude from training/validation
wildlife_insights_exclude_project_ids = {
    "backyard",
    "145625598251_2006485_550_it_ltser_2023",
}

# Wildlife Insights taxon ID for blank images
# see https://www.kaggle.com/models/google/speciesnet/keras?select=taxonomy_release.txt
species_id_blank = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6"

sat_feature_dims = dict(
    alphaearth=64,
    galileo=512,
    taxabind=512,
    dinov2=768,
    tessera=128,
    satbird=512,
    prithvi_v2=1280,
    dinov3=1024,
)

image_feature_dims = dict(
    dinov2_vits14=384,
    dinov2_vitb14=768,
    clip_vitbigg14=1280,
    speciesnet=1280,
    megadetector=1280,
    bioclip=512,
    dinov3_vitb16=768,
    dinov3_vitl16=1024,
    random_resnet50=2048,
    fully_random=512,
)

sat_dims = defaultdict(lambda: 768)
sat_dims.update(sat_feature_dims)

image_dims = defaultdict(lambda: 768)
image_dims.update(image_feature_dims)

default_image_backbone = "dinov2_vitb14"
default_sat_backbone = "alphaearth"

# maximum dimensionality to reduce features to via PCA
pca_dim = 128

# number of habitat descriptions to use for refitting species models
num_habitat_descriptions = 5

# whether to limit data to species range
limit_to_range = True

# neighborhood scale (in meters) for remote sensing image feature and covariate aggregation
rs_scale = 250

rs_model_kwargs = dict(
    satbird=dict(
        weights_path=str(
            weights_path / "satbird" / "satbird_rgbnir_epoch=38-step=26090.ckpt"
        )
    ),
    galileo=dict(weights_path=str(weights_path / "galileo" / "models" / "base")),
    dinov3=dict(
        weights_path=str(
            weights_path / "dinov3" / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        )
    ),
)

# geography
min_latitude = 24.396308
max_latitude = 49.384358
min_longitude = -125.0
max_longitude = -66.93457

# plotting config
scale = 1.22
fig_column_width = scale * 242 / 72.27  # in
fig_page_width = scale * 505.69374 / 72.27  # in
golden_ratio = (1 + 5**0.5) / 2
silver_ratio = 1 + 2**0.5
bronze_ratio = (3 + 13**0.5) / 2

# VisDiff config
visdiff_model_name = "gpt-5.2-2025-12-11"

focal_species_ids_v1 = [
    "5c7ce479-8a45-40b3-ae21-7c97dfae22f5",  # white-tailed deer
    "b57debc1-dff2-48d3-a400-f2b5021e71b0",  # northern raccoon
    "86f5b978-4f30-40cc-bd08-be9e3fba27a0",  # eastern gray squirrel
    "aaf3b049-36e6-46dd-9a07-8a580e9618b7",  # coyote
    "f67c4346-bbbd-43d5-9b19-b541d493019b",  # douglas' squirrel
    "febff896-db40-4ac8-bcfe-5bb99a600950",  # mule deer
    "87be3a5c-e60a-4e7e-88c7-21544914d067",  # virginia opossum
    "07843615-e1fc-49d8-9821-fd1d7ff2e773",  # nine-banded armadillo
    "ba76d46e-25de-45e2-90a8-bd279b650f7c",  # bobcat
    "667a4650-a141-4c4e-844e-58cdeaeb4ae1",  # eastern cottontail
    "e07c537d-08b8-4c31-9768-51e4adf5a5ab",  # eastern fox squirrel
    "1db1c6e2-2ea9-45a6-ab69-a730133298eb",  # eastern chipmunk
    "ac0e8ba7-7261-4d17-8645-11ed3d02165a",  # red fox
    "436ddfdd-bc43-44c3-a25d-34671d3430a0",  # american black bear
    "16ec4010-f175-4de7-8a99-85aadec3963b",  # western gray squirrel
    "f7fb32b6-1531-44e9-a7e2-a3197edafdb9",  # white-tailed antelope squirrel
]

interesting_species_ids = {
    "43320a08-bf31-49a5-8213-f032311c5765",  # american beaver
    "819ea972-9d43-40f8-a47d-f415c816de94",  # american marten
    "aaf3b049-36e6-46dd-9a07-8a580e9618b7",  # coyote
    "41930eb4-2283-445a-8758-1f0b2ff43cf9",  # eastern spotted skunk
    "febff896-db40-4ac8-bcfe-5bb99a600950",  # mule deer
    "00804e75-09ef-44e5-8984-85e365377d47",  # pronghorn
    "0f2e2c41-f1bb-4cdd-8e97-ba7cffba3e86",  # snowshoe hare
    "e70effd7-855e-4ba3-966c-a4fbb74eab13",  # american badger
    "df4c64cf-306d-4bf2-aa86-9e9c8d81fd41",  # long-tailed weasel
    "96fe1a07-7ef1-4a2f-99e1-ec2c9a78b532",  # great blue heron
}

denylist_species_ids = [
    "f2d233e3-80e3-433d-9687-e29ecc7a467a",  # mammal
    "e237b8d3-997f-4f3b-8736-8c3958f8182c",  # corvus species
    "5109acb4-e503-4147-a175-a3c6aa71f1e3",  # domestic horse
    "b1352069-a39c-4a84-a949-60044271c0c1",  # bird
    "9805e578-061a-49c0-8abc-ea135381c162",  # common raven
    "f1856211-cfb7-4a5b-9158-c0f72fd09ee6",  # blank
    "990ae9dd-7a59-4344-afcb-1b7b21368000",  # human
    "e2895ed5-780b-48f6-8a11-9e27cb594511",  # vehicle
    "3d80f1d6-b1df-4966-9ff4-94053c7a902a",  # domestic dog
]
