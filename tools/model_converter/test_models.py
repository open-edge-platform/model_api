#!/usr/bin/env python3
"""
Test script to run inference on all converted models.

Usage:
    python test_models.py --models-dir ./models --image path/to/image.jpg --output-dir ./outputs
"""

import argparse
import logging
from pathlib import Path

import cv2
from model_api.models import Model
from model_api.visualizer import Visualizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def find_all_models(models_dir: Path) -> list[Path]:
    """
    Find all model XML files in the models directory.

    Args:
        models_dir: Directory containing model folders

    Returns:
        List of paths to .xml model files
    """
    model_files = []
    
    # Find all .xml files in subdirectories
    for model_folder in models_dir.iterdir():
        if model_folder.is_dir():
            # Look for .xml files in each folder
            xml_files = list(model_folder.glob("*.xml"))
            model_files.extend(xml_files)
    
    return sorted(model_files)


def test_model(model_path: Path, image_path: Path, output_dir: Path, device: str = "AUTO") -> bool:
    """
    Test a single model with inference and visualization.

    Args:
        model_path: Path to the model XML file
        image_path: Path to the input image
        output_dir: Directory to save output images
        device: Device to run inference on

    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Extract model name (without extension)
    model_name = model_path.stem
    model_folder = model_path.parent.name
    
    # Determine model variant (fp16 or int8) from folder name
    if "int8" in model_folder.lower():
        variant = "int8"
    elif "fp16" in model_folder.lower():
        variant = "fp16"
    else:
        variant = "unknown"
    
    try:
        logger.info(f"Testing model: {model_folder}/{model_name}")
        
        # 1. Load model
        logger.debug(f"Loading model from: {model_path}")
        model = Model.create_model(str(model_path), device=device)
        
        # 2. Load image
        logger.debug(f"Loading image from: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        # 3. Run inference
        logger.debug("Running inference...")
        result = model(image)
        
        # 4. Visualize and save results
        logger.debug("Generating visualization...")
        vis = Visualizer().render(image, result)
        
        # Save with model name and variant
        output_path = output_dir / f"{model_name}_{variant}.jpg"
        cv2.imwrite(str(output_path), vis)
        logger.info(f"✓ Saved visualization to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to test model {model_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all converted models with inference and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing converted models (default: ./models)",
    )
    
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image for testing",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./test_outputs"),
        help="Directory to save visualization outputs (default: ./test_outputs)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="AUTO",
        help="Device to run inference on (default: AUTO)",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter models by name pattern (e.g., 'int8' or 'mobilenet')",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.models_dir.exists():
        logger.error(f"Models directory not found: {args.models_dir}")
        return 1
    
    if not args.image.exists():
        logger.error(f"Input image not found: {args.image}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Find all models
    logger.info(f"Searching for models in: {args.models_dir}")
    model_files = find_all_models(args.models_dir)
    
    if not model_files:
        logger.error("No models found")
        return 1
    
    # Filter models if requested
    if args.filter:
        model_files = [m for m in model_files if args.filter in str(m)]
        logger.info(f"Filtered to {len(model_files)} models matching '{args.filter}'")
    
    logger.info(f"Found {len(model_files)} model(s) to test")
    
    # Test each model
    successful = 0
    failed = 0
    
    for model_path in model_files:
        if test_model(model_path, args.image, args.output_dir, args.device):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Testing Summary:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {successful + failed}")
    logger.info("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
