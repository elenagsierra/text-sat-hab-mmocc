#!/usr/bin/env -S uv run --python 3.11 --script
#
# /// script
# dependencies = [
#     "mmocc",
#     "pyvisdiff",
#     "setuptools",
# ]
#
# [tool.uv.sources]
# mmocc = { path = "../.." }
# pyvisdiff = { git = "https://github.com/timmh/pyvisdiff.git" }
# ///
"""Run VisDiff using pre-generated local satellite imagery.

This entrypoint assumes satellite imagery already exists at:
  CACHE_PATH/sat_images/<loc_id>.png
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import fire


def _load_base_main():
    script_path = Path(__file__).with_name("08_visdiff.py")
    spec = spec_from_file_location("mmocc_sat_steps_08_visdiff_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base VisDiff script at {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Base VisDiff script missing main(): {script_path}")
    return module.main


main = _load_base_main()


if __name__ == "__main__":
    fire.Fire(main)
