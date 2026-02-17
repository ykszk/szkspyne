#!/usr/bin/env python3
"""
CLI script to apply random exposure simulation to all DICOM files in a directory.
"""

import argparse
import pydicom
import numpy as np
from pathlib import Path
import sys
from typing import Tuple
import json
import scipy.ndimage as ndi
from szkspyne.labelme_utils import LabelMe


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img)
    norm_img = (img - img_min) / (img_max - img_min)
    return norm_img


def simulate_exposure(img: np.ndarray, alpha: float, beta: float, gamma: float, max_value: int) -> np.ndarray:
    """Simulate exposure adjustment using a simple model."""
    adjusted = max_value * minmax_normalize(img)
    adjusted = alpha * (adjusted**gamma) + beta
    adjusted = np.clip(adjusted, 0, max_value)
    adjusted = max_value * minmax_normalize(adjusted)
    return adjusted


def simulate_similarity(img: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Simulate similarity transformation of the image by a given angle and scale."""
    if angle == 0:
        rotated = img
    else:
        rotated = ndi.rotate(img, angle, reshape=False, mode='nearest')
    if scale == 1:
        return rotated
    rotated = ndi.zoom(rotated, scale, order=3)
    return rotated


def simulate_gaussian_noise(img: np.ndarray, sigma: float, max_value: int) -> np.ndarray:
    """Add Gaussian noise to the image."""
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape)
    img = minmax_normalize(img)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    noisy_img = max_value * noisy_img
    return noisy_img


def random_simulation(
    img: np.ndarray,
    alpha_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    rotation_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    sigma_range: Tuple[float, float],
    max_value: int,
) -> np.ndarray:
    """Apply random exposure simulation to image."""
    alpha = np.random.uniform(*alpha_range)
    beta = np.random.uniform(*beta_range)
    gamma = np.random.uniform(*gamma_range)
    rotation = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    sigma = np.random.uniform(*sigma_range)
    print(
        f"Applying simulation with alpha={alpha:.3f}, beta={beta:.1f}, gamma={gamma:.3f}, rotation={rotation:.2f}, scale={scale:.2f}, sigma={sigma:.3f}"
    )
    img = simulate_similarity(img, rotation, scale)
    params = {"alpha": alpha, "beta": beta, "gamma": gamma, "rotation": rotation, "scale": scale, "sigma": sigma}
    img = simulate_exposure(img, alpha, beta, gamma, max_value)
    img = simulate_gaussian_noise(img, sigma, max_value)
    return img, params


def process_dicom_file(
    input_path: Path,
    output_path: Path,
    dynamic_range_bit: int,
    alpha_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    gamma_range: Tuple[float, float],
    rotation_range: Tuple[float, float],
    scale_range: Tuple[float, float],
    sigma_range: Tuple[float, float],
) -> bool:
    """Process a single DICOM file."""
    try:
        # Read DICOM file
        dcm = pydicom.dcmread(input_path)
        image = dcm.pixel_array

        # Normalize to specified dynamic range
        max_value = 2**dynamic_range_bit - 1
        normalized_image = max_value * minmax_normalize(image)

        # Apply random simulation
        simulated_image, params = random_simulation(
            normalized_image, alpha_range, beta_range, gamma_range, rotation_range, scale_range, sigma_range, max_value
        )

        # Update pixel array in DICOM
        # dcm.pixel_array.setflags(write=True)
        # dcm.pixel_array[:] = simulated_image.astype(dcm.pixel_array.dtype)
        dcm.PixelData = simulated_image.astype(dcm.pixel_array.dtype).tobytes()
        dcm.Rows, dcm.Columns = simulated_image.shape
        dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dcm.compress(pydicom.uid.RLELossless)

        # Save parameters to Comment field as JSON
        dcm.add_new((0x0020, 0x4000), 'LT', json.dumps(params))

        # Set window center and width
        dcm.WindowCenter = int(max_value / 2)
        dcm.WindowWidth = int(max_value)

        # Save the modified DICOM
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dcm.save_as(output_path)

        input_json_path = input_path.with_suffix('.json')
        if input_json_path.exists():
            output_json_path = output_path.with_suffix('.json')
            print(f"Processing associated LabelMe file {input_json_path.name} -> {output_json_path}")
            lm = LabelMe.from_file(input_json_path)
            lm.imagePath = str(output_path.resolve())
            lm = lm.rotate(-params["rotation"])
            lm = lm.scale(params["scale"])
            jsn = lm.model_dump_json()
            with open(output_path.with_suffix('.json'), 'w') as f:
                f.write(jsn)

        print(f"Processed: {input_path.name} -> {output_path.name}")
        return True

    except Exception as e:
        print(f"Error processing {input_path.name}: {e}")
        return False


def parse_range(range_str: str) -> Tuple[float, float]:
    """Parse a range string like '0.7,1.3' into a tuple."""
    try:
        parts = range_str.split(',')
        if len(parts) != 2:
            raise ValueError("Range must have exactly 2 values separated by comma")
        return (float(parts[0]), float(parts[1]))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid range format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply random exposure simulation to DICOM files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("input_dir", type=Path, help="Input directory containing DICOM files")
    parser.add_argument("output_dir", type=Path, help="Output directory for processed DICOM files")

    # Optional arguments with defaults
    parser.add_argument(
        "--dynamic-range-bit", type=int, default=12, help="Dynamic range in bits (determines max pixel value)"
    )
    parser.add_argument(
        "--alpha-range",
        type=parse_range,
        default="0.8,1.2",
        help="Alpha range for exposure simulation (format: min,max)",
    )
    parser.add_argument(
        "--beta-range",
        type=parse_range,
        default="-200,200",
        help="Beta range for exposure simulation (format: min,max)",
    )
    parser.add_argument(
        "--gamma-range",
        type=parse_range,
        default="0.9,1.1",
        help="Gamma range for exposure simulation (format: min,max)",
    )
    parser.add_argument(
        "--rotation-range",
        type=parse_range,
        default="-10,10",
        help="Rotation range in degrees for simulation (format: min,max)",
    )
    parser.add_argument(
        "--scale-range", type=parse_range, default="0.9,1.1", help="Scale range for simulation (format: min,max)"
    )
    parser.add_argument(
        "--sigma-range",
        type=parse_range,
        default="0,0",
        help="Sigma range for Gaussian noise (format: min,max)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results")
    parser.add_argument("--pattern", type=str, default="*.dcm", help="File pattern to match DICOM files")
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search for DICOM files in subdirectories"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)

    if not args.input_dir.is_dir():
        print(f"Error: Input path '{args.input_dir}' is not a directory")
        sys.exit(1)

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all DICOM files
    if args.recursive:
        dicom_files = list(args.input_dir.rglob(args.pattern))
    else:
        dicom_files = list(args.input_dir.glob(args.pattern))

    if not dicom_files:
        print(f"No DICOM files found matching pattern '{args.pattern}' in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(dicom_files)} DICOM files to process")
    print(f"Parameters:")
    print(f"  Dynamic range: {args.dynamic_range_bit} bits (max value: {2**args.dynamic_range_bit - 1})")
    print(f"  Alpha range: {args.alpha_range}")
    print(f"  Beta range: {args.beta_range}")
    print(f"  Gamma range: {args.gamma_range}")
    print()

    # Process each file
    success_count = 0
    for dicom_file in dicom_files:
        # Maintain relative directory structure
        relative_path = dicom_file.relative_to(args.input_dir)
        output_path = args.output_dir / relative_path
        assert isinstance(dicom_file, Path)
        if "243" not in str(dicom_file):
            continue

        if process_dicom_file(
            dicom_file,
            output_path,
            args.dynamic_range_bit,
            args.alpha_range,
            args.beta_range,
            args.gamma_range,
            args.rotation_range,
            args.scale_range,
            args.sigma_range,
        ):
            success_count += 1

    print(f"\nProcessing complete: {success_count}/{len(dicom_files)} files processed successfully")

    if success_count < len(dicom_files):
        sys.exit(1)


if __name__ == "__main__":
    main()
