"""
Example inference script for UFM (UniFlowMatch) models.
This script demonstrates how to load a pre-trained UFM model, predict correspondences
between two images, and visualize the results including flow output and covisibility mask.

Usage:
    python example_inference_enhanced.py --source examples/image_pairs/fire_academy_0.png --target examples/image_pairs/fire_academy_1.png
    python example_inference_enhanced.py --model refine --source img1.jpg --target img2.jpg --output results.png
"""

import argparse

import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch

from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence
from uniflowmatch.utils.viz import warp_image_with_flow


def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_correspondences(model, source_image, target_image):
    """Predict correspondences between source and target images."""
    with torch.no_grad():
        result = model.predict_correspondences_batched(
            source_image=torch.from_numpy(source_image),
            target_image=torch.from_numpy(target_image),
        )

        flow_output = result.flow.flow_output[0].cpu().numpy()
        covisibility = result.covisibility.mask[0].cpu().numpy()

    return flow_output, covisibility


def visualize_results(source_image, target_image, flow_output, covisibility, output_path="ufm_output.png"):
    """Create and save visualization of results."""
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: Input images and warped result
    axs[0, 0].imshow(source_image)
    axs[0, 0].set_title("Source Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(target_image)
    axs[0, 1].set_title("Target Image")
    axs[0, 1].axis("off")

    # Warp the image using flow
    warped_image = warp_image_with_flow(source_image, None, target_image, flow_output.transpose(1, 2, 0))
    warped_image = covisibility[..., None] * warped_image + (1 - covisibility[..., None]) * 255 * np.ones_like(
        warped_image
    )
    warped_image = np.clip(warped_image / 255.0, 0, 1)

    axs[0, 2].imshow(warped_image)
    axs[0, 2].set_title("Warped Source Image")
    axs[0, 2].axis("off")

    # Bottom row: Flow and covisibility visualizations
    flow_vis_image = flow_vis.flow_to_color(flow_output.transpose(1, 2, 0))
    axs[1, 0].imshow(flow_vis_image)
    axs[1, 0].set_title("Flow Visualization (Valid at Covisible Pixels)")
    axs[1, 0].axis("off")

    # Covisibility mask (thresholded)
    axs[1, 1].imshow(covisibility > 0.5, cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title("Covisibility Mask (>0.5)")
    axs[1, 1].axis("off")

    # Covisibility mask (continuous)
    heatmap = axs[1, 2].imshow(covisibility, cmap="viridis", vmin=0, vmax=1)
    axs[1, 2].set_title("Covisibility Confidence")
    axs[1, 2].axis("off")
    plt.colorbar(heatmap, ax=axs[1, 2], shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="UFM inference example")
    parser.add_argument(
        "--source", "-s", default="examples/image_pairs/fire_academy_0.png", help="Path to source image"
    )
    parser.add_argument(
        "--target", "-t", default="examples/image_pairs/fire_academy_1.png", help="Path to target image"
    )
    parser.add_argument("--model", choices=["base", "refine", "base-980", "refine-980", "base-dinov2l-init", "base-dinov2g-init"], default="base", help="Model variant to use")
    parser.add_argument("--output", "-o", default="ufm_output.png", help="Output visualization path")
    parser.add_argument("--show", action="store_true", help="Display the visualization")

    args = parser.parse_args()

    # Load model
    print(f"Loading UFM {args.model} model...")

    model_repo_map = {
        "base"              : "infinity1096/UFM-Base",
        "refine"            : "infinity1096/UFM-Refine",
        "base-980"          : "infinity1096/UFM-Base-980",
        "refine-980"        : "infinity1096/UFM-Refine-980",
        "base-dinov2l-init" : "infinity1096/UFM-Base-DINOv2L-init",
        "base-dinov2g-init" : "infinity1096/UFM-Base-DINOv2G-init"
    }

    if "base" in args.model:
        model = UniFlowMatchConfidence.from_pretrained(model_repo_map[args.model])
    elif "refine" in args.model:
        model = UniFlowMatchClassificationRefinement.from_pretrained(model_repo_map[args.model])
    else:
        raise ValueError("Please choose from [base, refine, base-980, refine-980, base-dinov2l-init, base-dinov2g-init]")

    model.eval()
    print("Model loaded successfully!")

    # Load and prepare images
    print(f"Loading images: {args.source}, {args.target}")
    source_image = load_image(args.source)
    target_image = load_image(args.target)

    print(f"Image shapes: {source_image.shape}, {target_image.shape}")

    # Predict correspondences
    print("Running inference...")
    flow_output, covisibility = predict_correspondences(model, source_image, target_image)

    # Visualize results
    fig = visualize_results(source_image, target_image, flow_output, covisibility, args.output)

    if args.show:
        plt.show()

    print("Inference completed!")


if __name__ == "__main__":
    main()
