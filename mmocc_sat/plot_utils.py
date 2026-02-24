import os
import urllib.request
import zipfile

import huez as hz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from mmocc_sat.config import cache_path, fig_column_width, golden_ratio


def setup_matplotlib():

    hz.use("scheme-1")

    plt.rcParams["font.family"] = "serif"

    font_dir = cache_path / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_font(slug: str, url: str):
        zip_path = font_dir / f"{slug}.zip"
        extract_dir = font_dir / slug

        if not zip_path.exists():
            print(f"Downloading {slug} font...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")

        if not extract_dir.exists():
            print(f"Extracting {slug} font files...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction complete.")

        return extract_dir

    font_targets = [
        (
            "computer-modern",
            "https://www.fontsquirrel.com/fonts/download/computer-modern",
        ),
        (
            "fontawesome-free-7.1.0-desktop",
            "https://use.fontawesome.com/releases/v7.1.0/fontawesome-free-7.1.0-desktop.zip",
        ),
    ]

    for slug, url in font_targets:
        extracted_dir = _ensure_font(slug, url)
        font_files = font_manager.findSystemFonts(fontpaths=[extracted_dir])
        for font_file in font_files:
            try:
                font_manager.fontManager.addfont(font_file)
            except OSError as exc:
                print(f"Skipping font {font_file}: {exc}")

    plt.rcParams["font.serif"] = [
        "CMU Serif" "Times",
        "DejaVu Serif",
        "serif",
    ]
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rc("axes", unicode_minus=False)
    plt.rcParams["font.size"] = 9.24991


setup_matplotlib()


def center_crop(img):
    # Center-crop to 16:9 aspect ratio while removing at least 10% from each side
    width, height = img.size

    # Calculate crop dimensions removing at least 10% from each side
    min_crop_width = width * 0.8  # Remove at least 10% from left and right
    min_crop_height = height * 0.8  # Remove at least 10% from top and bottom

    # Determine final crop size based on 16:9 aspect ratio
    target_aspect_ratio = 16 / 9

    if min_crop_width / min_crop_height > target_aspect_ratio:
        # Width is the limiting factor
        crop_height = min_crop_height
        crop_width = crop_height * target_aspect_ratio
    else:
        # Height is the limiting factor
        crop_width = min_crop_width
        crop_height = crop_width / target_aspect_ratio

    # Calculate crop coordinates for center crop
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    img = img.crop((left, top, right, bottom))

    return img


def plot_image_examples(
    image_paths,
    sat_paths,
    image_features,
    image_coefficients,
    sat_features,
    sat_coefficients,
    cov_features,
    cov_coefficients,
    species_name="",
    modalities="",
    image_backbone="",
):

    # Compute dot products for all images
    dot_products_image = np.dot(
        image_features, image_coefficients
    )  # Exclude intercept term
    dot_products_sat = np.dot(sat_features, sat_coefficients)  # Exclude intercept term
    dot_products_cov = np.dot(cov_features, cov_coefficients)  # Exclude intercept term

    for mode in ["image", "sat", "cov", "image_vs_sat", "image_vs_cov", "sat_vs_cov"]:

        # Get indices for top 10 and bottom 10
        if mode == "image":
            dot_products = dot_products_image
        elif mode == "sat":
            dot_products = dot_products_sat
        elif mode == "cov":
            dot_products = dot_products_cov
        elif mode == "image_vs_sat":
            dot_products = dot_products_image - dot_products_sat
        elif mode == "image_vs_cov":
            dot_products = dot_products_image - dot_products_cov
        elif mode == "sat_vs_cov":
            dot_products = dot_products_sat - dot_products_cov
        else:
            raise ValueError(f"Unknown mode: {mode}")

        top_indices = np.argsort(dot_products)[-10:][::-1]  # Top 10 in descending order
        bottom_indices = np.argsort(dot_products)[:10]  # Bottom 10 in ascending order

        # Create subplots - now with 2 columns per example (image + satellite)
        fig, axes = plt.subplots(4, 10, figsize=(40, 16))
        fig.suptitle(f"{species_name} - {modalities} - {image_backbone}", fontsize=16)

        # Plot top 10 images with corresponding satellite images
        for i, idx in enumerate(top_indices):
            row = i // 5
            img_col = (i % 5) * 2
            sat_col = img_col + 1

            # Plot camera trap image
            try:
                img = Image.open(image_paths[idx])
                img = center_crop(img)
                axes[row, img_col].imshow(img)
                axes[row, img_col].set_title(
                    f"Top {i+1} Camera\nScore: {dot_products[idx]:.3f}", fontsize=10
                )
                axes[row, img_col].axis("off")
            except Exception as e:
                axes[row, img_col].text(
                    0.5,
                    0.5,
                    f"Error loading\n{str(e)[:30]}...",
                    ha="center",
                    va="center",
                    transform=axes[row, img_col].transAxes,
                )
                axes[row, img_col].set_title(
                    f"Top {i+1} Camera\nScore: {dot_products[idx]:.3f}", fontsize=10
                )
                axes[row, img_col].axis("off")

            # Plot corresponding satellite image
            try:
                if sat_paths is not None and idx < len(sat_paths):
                    sat_img = Image.open(sat_paths[idx])
                    axes[row, sat_col].imshow(sat_img)
                    axes[row, sat_col].set_title(f"Top {i+1} Satellite", fontsize=10)
                    axes[row, sat_col].axis("off")
                else:
                    axes[row, sat_col].text(
                        0.5,
                        0.5,
                        "No satellite\nimage available",
                        ha="center",
                        va="center",
                        transform=axes[row, sat_col].transAxes,
                    )
                    axes[row, sat_col].set_title(f"Top {i+1} Satellite", fontsize=10)
                    axes[row, sat_col].axis("off")
            except Exception as e:
                axes[row, sat_col].text(
                    0.5,
                    0.5,
                    f"Error loading\n{str(e)[:30]}...",
                    ha="center",
                    va="center",
                    transform=axes[row, sat_col].transAxes,
                )
                axes[row, sat_col].set_title(f"Top {i+1} Satellite", fontsize=10)
                axes[row, sat_col].axis("off")

        # Plot bottom 10 images with corresponding satellite images
        for i, idx in enumerate(bottom_indices):
            row = (i // 5) + 2  # Start from row 2
            img_col = (i % 5) * 2
            sat_col = img_col + 1

            # Plot camera trap image
            try:
                img = Image.open(image_paths[idx])
                img = center_crop(img)
                axes[row, img_col].imshow(img)
                axes[row, img_col].set_title(
                    f"Bottom {i+1} Camera\nScore: {dot_products[idx]:.3f}", fontsize=10
                )
                axes[row, img_col].axis("off")
            except Exception as e:
                axes[row, img_col].text(
                    0.5,
                    0.5,
                    f"Error loading\n{str(e)[:30]}...",
                    ha="center",
                    va="center",
                    transform=axes[row, img_col].transAxes,
                )
                axes[row, img_col].set_title(
                    f"Bottom {i+1} Camera\nScore: {dot_products[idx]:.3f}", fontsize=10
                )
                axes[row, img_col].axis("off")

            # Plot corresponding satellite image
            try:
                if sat_paths is not None and idx < len(sat_paths):
                    sat_img = Image.open(sat_paths[idx])
                    axes[row, sat_col].imshow(sat_img)
                    axes[row, sat_col].set_title(f"Bottom {i+1} Satellite", fontsize=10)
                    axes[row, sat_col].axis("off")
                else:
                    axes[row, sat_col].text(
                        0.5,
                        0.5,
                        "No satellite\nimage available",
                        ha="center",
                        va="center",
                        transform=axes[row, sat_col].transAxes,
                    )
                    axes[row, sat_col].set_title(f"Bottom {i+1} Satellite", fontsize=10)
                    axes[row, sat_col].axis("off")
            except Exception as e:
                axes[row, sat_col].text(
                    0.5,
                    0.5,
                    f"Error loading\n{str(e)[:30]}...",
                    ha="center",
                    va="center",
                    transform=axes[row, sat_col].transAxes,
                )
                axes[row, sat_col].set_title(f"Bottom {i+1} Satellite", fontsize=10)
                axes[row, sat_col].axis("off")

        plt.tight_layout()
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "figures", "image_examples_interactions"
        )
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir,
                f"image_examples_{mode}_{species_name.lower().replace(' ', '-')}_{modalities}_{image_backbone}.jpg",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(
                output_dir,
                f"image_examples_{mode}_{species_name.lower().replace(' ', '-')}_{modalities}_{image_backbone}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close()
