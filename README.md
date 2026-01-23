# Deep Multi-modal Species Occupancy Modeling

This codebase accompanies [our paper](https://doi.org/10.1101/2025.09.06.674602) on species occupancy modeling using traditional environmenal variables, satellite imagery, and blank camera trap backgrounds.

## Downloading models and predicting occupancy

We provide a minimal demo notebook in `mmocc/analysis/occupancy_prediction_demo.ipynb` that demonstrates how to generate occupancy predictions on a new site using one of our fitted models. To run this notebook, follow these steps download [occupancy_prediction_demo.ipynb](mmocc/analysis/occupancy_prediction_demo.ipynb) and open in your favorite Jupyter notebook editor or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timmh/mmocc/blob/main/mmocc/analysis/occupancy_prediction_demo.ipynb) and run all cells in order. The notebook will automatically install required dependencies, download fitted model weights,and make an occupancy probability prediction based on a new camera trap image.

## Running the full experimental pipeline
1. Download the code and `cd` into it
2. Download weights for [DINOv3](https://github.com/facebookresearch/dinov3), [Galileo](https://github.com/nasaharvest/galileo), [SatBird](https://github.com/RolnickLab/SatBird), and [SpeciesNet](https://github.com/google/cameratrapai), and place them into a directory of your choice. See [this section](#weights) for how to organize that directory.
3. Copy `.env.example` to `.env`, then customize `.env` to your environment and filesystem paths.
4. Download camera trap metadata and images from [Wildlife Insights](https://app.wildlifeinsights.org). The specific datasets used in this work can be found in the data references of our paper.
5. As an alternative to downloading large volumes of camera trap data and performing feature extraction and model fitting yourself, you can download our intermediate results [here](https://data.csail.mit.edu/mmocc/mmocc_cache.tar.gz) and set your `CACHE_PATH` to the extracted directory.
6. For simplicity and reproducibility, we recommend using [uv](https://docs.astral.sh/uv/) for running code. This allows having separate, minimal environments for each step and strictly defined versions of dependencies. Once you have `uv` installed, simply run `mmocc/steps/$STEP` to execute an individual pipeline step or `for step in mmocc/steps/*.py; do $step; done` to run all steps consecutively.
7. Alternatively, you can try installing all dependencies in a single Python environment by activating your environment and running `pip install -e ".[all]"`. Note that this might lead to conflicting dependencies. Then run `mmocc/steps/$STEP` to execute an individual pipeline step or `for step in mmocc/steps/*.py; do python $step; done` to run all steps consecutively.
8. Run the scripts and Jupyter notebooks in `mmocc/analysis` to print results and export figures.

## Weights
Your final weights directory should look as follows:
```
weights
├── dinov3
│   ├── dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth
│   ├── dinov3_vit7b16_pretrain_sat493m-a6675841.pth
│   ├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
│   ├── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
│   ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
│   └── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
├── galileo
│   └── models
│       ├── base
│       │   ├── config.json
│       │   ├── decoder.pt
│       │   ├── encoder.pt
│       │   ├── second_decoder.pt
│       │   └── target_encoder.pt
│       ├── nano
│       │   ├── config.json
│       │   ├── decoder.pt
│       │   ├── encoder.pt
│       │   ├── second_decoder.pt
│       │   └── target_encoder.pt
│       └── tiny
│           ├── config.json
│           ├── decoder.pt
│           ├── encoder.pt
│           ├── second_decoder.pt
│           └── target_encoder.pt
├── satbird
│   └── satbird_rgbnir_epoch=38-step=26090.ckpt
└── speciesnet_4.0.1b
    ├── full_image_88545560_22x8_v12_epoch_00153.labels.txt
    ├── full_image_88545560_22x8_v12_epoch_00153.pt
    ├── geofence_release.2025.02.27.0702.json
    ├── info.json
    ├── README.md
    └── taxonomy_release.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue or email [haucke@mit.edu](mailto:haucke@mit.edu).

## Acknowledgements
This work was supported by the AI and Biodiversity Change (ABC) Global Center, which is funded by the [US National Science Foundation under Award No. 2330423](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2330423&HistoricalAwards=false) and [Natural Sciences and Engineering Research Council of Canada under Award No. 585136](https://www.nserc-crsng.gc.ca/ase-oro/Details-Detailles_eng.asp?id=782440). This work draws on research supported in part by the Social Sciences and Humanities Research Council.

## Citing
If you find this repository useful, please consider citing:
```
@article{mmocc,
  title={Deep Multi-modal Species Occupancy Modeling},
  author={Haucke, Timm and Harrell, Lauren and Shen, Yunyi and Klein, Levente and Rolnick, David and Gillespie, Lauren and Beery, Sara},
  journal={bioRxiv},
  pages={2025--09},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```