#!/usr/bin/env python3
"""
Export timm models from HuggingFace to OpenVINO format.

This tool downloads timm models from HuggingFace and exports them to OpenVINO IR format
with support for multiple precision formats (FP32, FP16, INT8 weight-only quantization).
All model metadata (input size, preprocessing parameters) is auto-detected from timm's
pretrained_cfg.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import timm
import torch
from optimum.intel import OVModelForImageClassification


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles export of timm models to OpenVINO format."""
    
    SUPPORTED_FORMATS = ['fp32', 'fp16', 'int8-wo']
    DEFAULT_FORMATS = ['fp32', 'fp16', 'int8-wo']
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        batch_size: int = 1,
        weight_format: Optional[str] = None
    ):
        """
        Initialize the ModelExporter.
        
        Args:
            config_path: Path to config.json file
            output_dir: Output directory for exported models
            batch_size: Batch size for model export (default: 1)
            weight_format: Specific weight format to export. If None, export all formats.
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.weight_format = weight_format
        
        # Determine which formats to export
        if weight_format
            if weight_format not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported weight format: {weight_format}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            self.formats_to_export = [weight_format]
        else:
            self.formats_to_export = self.DEFAULT_FORMATS
        
        # Track export results
        self.results = {
            'successful': [],
            'failed': []
        }
    
    def load_config(self) -> List[Dict]:
        """Load and validate configuration file."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        if not isinstance(config, list):
            raise ValueError("Configuration must be a JSON array")
        
        # Deduplicate repos
        seen_repos = set()
        unique_config = []
        
        for entry in config:
            if 'huggingface_repo' not in entry:
                logger.warning(f"Skipping entry without 'huggingface_repo' field: {entry}")
                continue
            
            repo = entry['huggingface_repo']
            
            if repo in seen_repos:
                logger.warning(f"Skipping duplicate repository: {repo}")
                continue
            
            seen_repos.add(repo)
            unique_config.append(entry)
        
        logger.info(f"Loaded {len(unique_config)} unique models from config")
        return unique_config
    
    def extract_model_name(self, huggingface_repo: str) -> str:
        """
        Extract model short name from HuggingFace repo path.
        
        Args:
            huggingface_repo: HuggingFace repo path (e.g., 'timm/mobilenetv2_050.lamb_in1k')
        
        Returns:
            Model short name (e.g., 'mobilenetv2_050.lamb_in1k')
        """
        # Extract name after the last slash
        model_name = huggingface_repo.split('/')[-1]
        return model_name
    
    def get_model_metadata(self, model: torch.nn.Module) -> Dict:
        """
        Extract metadata from timm model's pretrained_cfg.
        
        Args:
            model: Loaded timm model
        
        Returns:
            Dictionary containing model metadata
        """
        cfg = model.pretrained_cfg
        
        # Extract input size (C, H, W)
        input_size = cfg.input_size if hasattr(cfg, 'input_size') else (3, 224, 224)
        
        # Extract preprocessing parameters
        mean = list(cfg.mean) if hasattr(cfg, 'mean') else [0.485, 0.456, 0.406]
        std = list(cfg.std) if hasattr(cfg, 'std') else [0.229, 0.224, 0.225]
        
        # Additional metadata
        metadata = {
            'input_size': list(input_size),
            'input_shape': [self.batch_size] + list(input_size),
            'mean': mean,
            'std': std,
            'num_classes': cfg.num_classes if hasattr(cfg, 'num_classes') else 1000,
            'interpolation': cfg.interpolation if hasattr(cfg, 'interpolation') else 'bicubic',
            'crop_pct': cfg.crop_pct if hasattr(cfg, 'crop_pct') else 0.875,
            'batch_size': self.batch_size,
        }
        
        return metadata
    
    def export_model(
        self,
        huggingface_repo: str,
        model_short_name: str,
        precision: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Export a single model with specified precision.
        
        Args:
            huggingface_repo: HuggingFace repository path
            model_short_name: Short name for the model
            precision: Precision format (fp32, fp16, int8-wo)
        
        Returns:
            Tuple of (success, output_path, error_message)
        """
        output_name = f"{model_short_name}-{precision}"
        output_path = self.output_dir / output_name
        
        try:
            logger.info(f"Exporting {model_short_name} to {precision}...")
            
            # First, load the timm model to extract metadata
            logger.info(f"Loading timm model: {huggingface_repo}")
            timm_model = timm.create_model(huggingface_repo, pretrained=True)
            metadata = self.get_model_metadata(timm_model)
            
            # Clean up timm model to free memory
            del timm_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Configure export parameters based on precision
            export_kwargs = {
                'export': True,
                'library': 'timm',
            }
            
            # Map precision to weight_format parameter
            if precision == 'fp16':
                export_kwargs['half'] = True
            elif precision == 'int8-wo':
                # Weight-only INT8 quantization
                export_kwargs['compression_option'] = 'int8'
            # fp32 is the default, no special parameters needed
            
            # Export using optimum-intel
            logger.info(f"Converting to OpenVINO IR format: {precision}")
            ov_model = OVModelForImageClassification.from_pretrained(
                huggingface_repo,
                **export_kwargs
            )
            
            # Save the model
            logger.info(f"Saving to {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
            ov_model.save_pretrained(output_path)
            
            # Save metadata
            metadata_path = output_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Successfully exported {output_name}")
            return True, str(output_path), None
            
        except Exception as e:
            error_msg = f"Failed to export {output_name}: {str(e)}"
            logger.error(f"✗ {error_msg}")
            return False, None, error_msg
    
    def export_all(self):
        """Export all models from configuration."""
        config = self.load_config()
        
        logger.info(f"Starting export process")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Formats to export: {', '.join(self.formats_to_export)}")
        logger.info(f"Total models: {len(config)}")
        logger.info("-" * 80)
        
        start_time = datetime.now()
        total_exports = len(config) * len(self.formats_to_export)
        completed_exports = 0
        
        for idx, entry in enumerate(config, 1):
            huggingface_repo = entry['huggingface_repo']
            model_short_name = self.extract_model_name(huggingface_repo)
            
            logger.info(f"\n[{idx}/{len(config)}] Processing: {model_short_name}")
            logger.info(f"Repository: {huggingface_repo}")
            
            # Export each format
            for precision in self.formats_to_export:
                success, output_path, error_msg = self.export_model(
                    huggingface_repo,
                    model_short_name,
                    precision
                )
                
                completed_exports += 1
                
                if success:
                    self.results['successful'].append({
                        'model': model_short_name,
                        'repo': huggingface_repo,
                        'precision': precision,
                        'output_path': output_path
                    })
                else:
                    self.results['failed'].append({
                        'model': model_short_name,
                        'repo': huggingface_repo,
                        'precision': precision,
                        'error': error_msg
                    })
                
                logger.info(f"Progress: {completed_exports}/{total_exports} exports completed")
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPORT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total time: {duration}")
        logger.info(f"Successful exports: {len(self.results['successful'])}/{total_exports}")
        logger.info(f"Failed exports: {len(self.results['failed'])}/{total_exports}")
        
        if self.results['successful']:
            logger.info("\n✓ SUCCESSFUL EXPORTS:")
            for item in self.results['successful']:
                logger.info(f"  - {item['model']}-{item['precision']}: {item['output_path']}")
        
        if self.results['failed']:
            logger.info("\n✗ FAILED EXPORTS:")
            for item in self.results['failed']:
                logger.info(f"  - {item['model']}-{item['precision']}: {item['error']}")
        
        # Save results to JSON
        results_path = self.output_dir / 'export_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'config': {
                    'batch_size': self.batch_size,
                    'formats': self.formats_to_export,
                    'total_models': len(config),
                },
                'results': self.results
            }, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {results_path}")
        logger.info("=" * 80)
        
        # Return exit code
        return 0 if not self.results['failed'] else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export timm models from HuggingFace to OpenVINO format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all models with all formats (fp32, fp16, int8-wo)
  python export_models.py
  
  # Export only FP16 models
  python export_models.py --weight-format fp16
  
  # Export with batch size 4
  python export_models.py --batch-size 4
  
  # Custom config and output directory
  python export_models.py --config models.json --output-dir ./exported_models
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration JSON file (default: config.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for exported models (default: ./output)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for model export (default: 1)'
    )
    
    parser.add_argument(
        '--weight-format',
        type=str,
        choices=['fp32', 'fp16', 'int8-wo'],
        help='Export only this weight format. If not specified, exports all formats (fp32, fp16, int8-wo)'
    )
    
    args = parser.parse_args()
    
    try:
        exporter = ModelExporter(
            config_path=args.config,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            weight_format=args.weight_format
        )
        
        exit_code = exporter.export_all()
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
