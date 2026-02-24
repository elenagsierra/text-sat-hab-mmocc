from functools import cache

import ee
import numpy as np

from mmocc_sat.config import rs_scale
from mmocc_sat.utils import memory


@cache
def initialize_ee():
    ee.Initialize()


@memory.cache
def get_covariates(longitude, latitude, covariates):
    initialize_ee()

    bioclim = ee.Image("WORLDCLIM/V1/BIO")  # type: ignore

    orcdrc = (
        ee.Image("projects/soilgrids-isric/ocd_mean")  # type: ignore
        .select("ocd_15-30cm_mean")
        .rename("orcdrc")
    )
    phihox = (
        ee.Image("projects/soilgrids-isric/phh2o_mean")  # type: ignore
        .select("phh2o_15-30cm_mean")
        .rename("phihox")
    )
    cecsol = (
        ee.Image("projects/soilgrids-isric/cec_mean")  # type: ignore
        .select("cec_15-30cm_mean")
        .rename("cecsol")
    )
    # bdticm = None
    clyppt = (
        ee.Image("projects/soilgrids-isric/clay_mean")  # type: ignore
        .select("clay_15-30cm_mean")
        .rename("clyppt")
    )
    sltppt = (
        ee.Image("projects/soilgrids-isric/silt_mean")  # type: ignore
        .select("silt_15-30cm_mean")
        .rename("sltppt")
    )
    sndppt = (
        ee.Image("projects/soilgrids-isric/sand_mean")  # type: ignore
        .select("sand_15-30cm_mean")
        .rename("sndppt")
    )
    bldfie = (
        ee.Image("projects/soilgrids-isric/bdod_mean")  # type: ignore
        .select("bdod_15-30cm_mean")
        .rename("bldfie")
    )

    soilgrids_satbird = ee.Image.cat(  # type: ignore
        [
            orcdrc,
            phihox,
            cecsol,
            # bdticm,
            clyppt,
            sltppt,
            sndppt,
            bldfie,
        ]
    )

    full_image = ee.Image.cat(  # type: ignore
        [
            bioclim,
            soilgrids_satbird,
        ]
    )

    point = ee.Geometry.Point([longitude, latitude])  # type: ignore
    feature = ee.Feature(point)  # type: ignore
    fc = ee.FeatureCollection([feature])  # type: ignore

    # Use reduceRegions to get the mean of pixels in the region, ignoring masked pixels.
    # This prevents a single masked pixel from nullifying the result for a band.
    sampled = full_image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),  # type: ignore
        scale=rs_scale,
    )

    info = sampled.getInfo()
    all_features = info["features"]  # type: ignore

    if len(all_features) == 0:
        print(f"No features found for point ({longitude}, {latitude})")
        return np.full((len(covariates),), np.nan).astype(np.float32)

    properties = all_features[0]["properties"]
    return np.array([properties.get(key) for key in covariates]).astype(np.float32)
