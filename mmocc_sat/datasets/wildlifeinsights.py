import pandas as pd
import torch

from mmocc.config import cache_path, default_covariates, wi_image_path
from mmocc.rs_utils import get_covariates
from mmocc.utils import load_image


class WildlifeInsightsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.df = pd.read_pickle(cache_path / "wi_blank_images.pkl")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        loc_id = row["loc_id"]
        datetime_posix = row["Date_Time"].to_pydatetime().timestamp()
        image_path = row["FilePath"].replace("gs://", str(wi_image_path) + "/")

        image = load_image(image_path)

        covariates = get_covariates(
            row["Longitude"],
            row["Latitude"],
            covariates=default_covariates,
        )

        return dict(
            loc_id=loc_id,
            datetime=datetime_posix,
            latitude=row["Latitude"],
            longitude=row["Longitude"],
            image=image,
            image_path=image_path,
            covariates=covariates,
        )
