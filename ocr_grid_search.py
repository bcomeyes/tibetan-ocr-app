# %% [markdown]
# # Tibetan OCR Grid Search with Quality Scoring
# 
# This notebook systematically tests OCR parameter combinations across different
# Tibetan script types (Uchen/Umeh) and quality levels to find optimal settings.
#
# **NEW: Automated OCR Quality Scoring with PyBo**
# - Each OCR output is automatically scored using PyBo tokenization
# - Score represents % of valid Tibetan words (0-100)
# - Results are ranked by quality score
# - Top performers can be reviewed first, skipping garbage outputs
#
# ## Design Decisions
#
# **Model-to-Script Matching:**
# - Uchen samples → Woodblock, Woodblock-Stacks, Modern
# - Umeh samples → Ume_Druma, Ume_Petsuk, Modern
#
# **Parameters Tested:**
# - ocr_model_name: Which OCR model (matched to script type)
# - line_mode: "line" or "layout" detection
# - k_factor: Line extraction expansion [2.0, 2.5, 3.0]
# - bbox_tolerance: Bounding box merge tolerance [2.5, 3.5, 4.0, 5.0]
# - merge_lines: Whether to merge line chunks [True, False]
# - tps_threshold: Dewarping sensitivity [0.1, 0.25, 0.5, 0.9]
# - class_threshold: Line detection confidence [0.7, 0.8, 0.9]
#
# ## OCR Quality Scoring
#
# After each OCR run, PyBo tokenizes the output text and calculates:
# - **Quality Score**: Percentage of tokens that are valid Tibetan words (0-100)
# - **High scores (>90)**: Likely excellent OCR output
# - **Medium scores (50-90)**: Mixed quality
# - **Low scores (<50)**: Likely garbage output
#
# This allows automatic filtering: instead of manually reviewing 1,728 outputs,
# you can focus on the top 50-100 highest-scoring results.

# %% Imports and Path Setup
import os
import sys
import cv2
import json
import signal
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import time
import traceback

# PyBo for OCR quality scoring
try:
    from pybo import WordTokenizer
    PYBO_AVAILABLE = True
    print("✅ PyBo loaded for OCR quality scoring")
except ImportError:
    PYBO_AVAILABLE = False
    print("⚠️  PyBo not available - quality scoring disabled")
    print("   Install with: pip install git+https://github.com/OpenPecha/pybo.git")

# Add the project root to path
PROJECT_ROOT = Path(__file__).parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Now import BDRC modules
from BDRC.Data import (
    Encoding, LineMode, TPSMode, Platform,
    LineDetectionConfig, LayoutDetectionConfig, OCRModelConfig
)
from BDRC.Inference import OCRPipeline
from BDRC.Utils import import_local_models, get_platform
from BDRC.utils.pdf_extract import extract_images_from_pdf

print(f"Project root: {PROJECT_ROOT}")
print(f"Platform: {get_platform()}")

# %% Configuration - EDIT THESE PATHS
"""
=============================================================================
CONFIGURATION - Edit these paths to match your local setup
=============================================================================
"""

# Base directory for the tibetan-ocr-app
BASE_DIR = Path.home() / "Documents" / "tibetan-ocr-app"

# Model paths
OCR_MODELS_DIR = BASE_DIR / "OCRModels"
LINE_MODEL_PATH = BASE_DIR / "Models" / "Lines" / "PhotiLines.onnx"
LAYOUT_MODEL_PATH = BASE_DIR / "Models" / "Layout" / "photi.onnx"

# Test samples directory
TEST_SAMPLES_DIR = BASE_DIR / "input_files"

# Output directory for results
OUTPUT_DIR = BASE_DIR / "grid_search_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Temp directory for extracted PDF images
TEMP_DIR = OUTPUT_DIR / "temp_images"
TEMP_DIR.mkdir(exist_ok=True)

# Checkpoint directory
CHECKPOINT_DIR = OUTPUT_DIR / "_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = CHECKPOINT_DIR / "progress.json"

print(f"OCR Models: {OCR_MODELS_DIR}")
print(f"Test Samples: {TEST_SAMPLES_DIR}")
print(f"Output: {OUTPUT_DIR}")

# %% Initialize PyBo Quality Scorer
class OCRQualityScorer:
    """
    Score OCR text quality using PyBo tokenization.
    
    Higher scores = more valid Tibetan words recognized by THL lexicon.
    """
    
    def __init__(self):
        if not PYBO_AVAILABLE:
            self.tokenizer = None
            print("⚠️  Quality scoring disabled (PyBo not available)")
        else:
            self.tokenizer = WordTokenizer()
            print("✅ Quality scorer initialized with PyBo")
    
    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Score OCR text quality.
        
        Returns:
            dict with keys:
                - quality_score: 0-100, percentage of valid words
                - total_tokens: number of word tokens
                - valid_tokens: number recognized in lexicon
                - invalid_tokens: number not recognized
        """
        if not self.tokenizer or not text.strip():
            return {
                'quality_score': 0.0,
                'total_tokens': 0,
                'valid_tokens': 0,
                'invalid_tokens': 0
            }
        
        try:
            tokens = self.tokenizer.tokenize(text)
            
            total_tokens = 0
            valid_tokens = 0
            invalid_tokens = 0
            
            for token in tokens:
                # Get attributes safely
                pos = getattr(token, 'pos', None)
                
                # Skip punctuation (empty POS)
                if pos == '':
                    continue
                
                total_tokens += 1
                
                # Valid if has real POS tag (not NON_WORD, NO_POS, OTHER, etc.)
                if pos and pos not in ['NON_WORD', 'non-word', 'NO_POS', 'OTHER', '', None, 'X']:
                    valid_tokens += 1
                else:
                    invalid_tokens += 1
            
            quality_score = (valid_tokens / total_tokens * 100) if total_tokens > 0 else 0.0
            
            return {
                'quality_score': round(quality_score, 2),
                'total_tokens': total_tokens,
                'valid_tokens': valid_tokens,
                'invalid_tokens': invalid_tokens
            }
            
        except Exception as e:
            print(f"  ⚠️  Quality scoring error: {e}")
            return {
                'quality_score': 0.0,
                'total_tokens': 0,
                'valid_tokens': 0,
                'invalid_tokens': 0
            }

# Initialize global scorer
quality_scorer = OCRQualityScorer()

# %% Verify paths exist
def verify_setup():
    """Verify all required paths and models exist."""
    errors = []
    
    if not OCR_MODELS_DIR.exists():
        errors.append(f"OCR Models directory not found: {OCR_MODELS_DIR}")
    
    if not LINE_MODEL_PATH.exists():
        errors.append(f"Line detection model not found: {LINE_MODEL_PATH}")
    
    if not LAYOUT_MODEL_PATH.exists():
        errors.append(f"Layout detection model not found: {LAYOUT_MODEL_PATH}")
    
    if not TEST_SAMPLES_DIR.exists():
        errors.append(f"Test samples directory not found: {TEST_SAMPLES_DIR}")
    
    if errors:
        print("❌ Setup errors:")
        for e in errors:
            print(f"   - {e}")
        return False
    
    # List available OCR models
    ocr_models = list(OCR_MODELS_DIR.iterdir())
    print(f"✅ Found {len(ocr_models)} OCR models:")
    for m in ocr_models:
        if m.is_dir():
            print(f"   - {m.name}")
    
    # List test sample files (JSON or PDF)
    test_files = list(TEST_SAMPLES_DIR.glob("*.json")) + list(TEST_SAMPLES_DIR.glob("*.pdf"))
    print(f"\n✅ Found {len(test_files)} test files:")
    for f in test_files[:10]:  # Show first 10
        print(f"   - {f.name}")
    if len(test_files) > 10:
        print(f"   ... and {len(test_files) - 10} more")
    
    return True

verify_setup()

# %% Define Parameter Grid
"""
=============================================================================
PARAMETER GRID DEFINITION
=============================================================================
"""

@dataclass
class GridSearchParams:
    """Parameters for a single grid search run."""
    # OCR Model
    ocr_model_name: str
    
    # Line detection
    line_mode: str  # "line" or "layout"
    class_threshold: float  # Confidence threshold for detection
    
    # Line processing
    k_factor: float  # Line extraction expansion
    bbox_tolerance: float  # BBox merge tolerance
    merge_lines: bool  # Merge line chunks
    
    # Dewarping (use_tps always True, threshold controls sensitivity)
    tps_threshold: float  # 0.9 = effectively off, 0.1 = aggressive
    
    def to_filename(self) -> str:
        """Generate a descriptive filename from parameters."""
        merge_str = "T" if self.merge_lines else "F"
        return (
            f"{self.ocr_model_name}_{self.line_mode}_"
            f"k{self.k_factor}_bbox{self.bbox_tolerance}_"
            f"merge-{merge_str}_tps{self.tps_threshold}_conf{self.class_threshold}"
        )


# Models for testing (will auto-detect from input files)
# Assuming all Tengyur files are Uchen for now
MODELS_TO_TEST = ["Woodblock", "Woodblock-Stacks", "Modern"]

# Parameter values to test
PARAM_VALUES = {
    "line_mode": ["line", "layout"],
    "class_threshold": [0.7, 0.8, 0.9],
    "k_factor": [2.0, 2.5, 3.0],
    "bbox_tolerance": [2.5, 3.5, 4.0, 5.0],
    "merge_lines": [True, False],
    "tps_threshold": [0.1, 0.25, 0.5, 0.9],
}


def generate_param_combinations() -> List[GridSearchParams]:
    """Generate all parameter combinations."""
    
    combinations = []
    
    for model in MODELS_TO_TEST:
        for line_mode in PARAM_VALUES["line_mode"]:
            for class_threshold in PARAM_VALUES["class_threshold"]:
                for k_factor in PARAM_VALUES["k_factor"]:
                    for bbox_tolerance in PARAM_VALUES["bbox_tolerance"]:
                        for merge_lines in PARAM_VALUES["merge_lines"]:
                            for tps_threshold in PARAM_VALUES["tps_threshold"]:
                                combinations.append(GridSearchParams(
                                    ocr_model_name=model,
                                    line_mode=line_mode,
                                    class_threshold=class_threshold,
                                    k_factor=k_factor,
                                    bbox_tolerance=bbox_tolerance,
                                    merge_lines=merge_lines,
                                    tps_threshold=tps_threshold,
                                ))
    
    return combinations


def calculate_combinations_count() -> int:
    """Calculate total combinations."""
    total = len(MODELS_TO_TEST)
    for values in PARAM_VALUES.values():
        total *= len(values)
    return total


# Show combination counts
total_combos = calculate_combinations_count()
print(f"\nParameter combinations per image: {total_combos}")

# %% Checkpoint Management
class CheckpointManager:
    """Manages saving and loading progress for resumable grid search."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.completed = self._load()
    
    def _load(self) -> set:
        """Load completed image paths from checkpoint file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get("completed_images", []))
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                return set()
        return set()
    
    def save(self):
        """Save current progress to checkpoint file."""
        data = {
            "completed_images": list(self.completed),
            "last_updated": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def mark_completed(self, image_path: str):
        """Mark an image as fully processed (all param combinations done)."""
        self.completed.add(image_path)
        self.save()
    
    def is_completed(self, image_path: str) -> bool:
        """Check if an image has already been fully processed."""
        return image_path in self.completed
    
    def get_completed_count(self) -> int:
        """Get number of completed images."""
        return len(self.completed)
    
    def reset(self):
        """Clear all checkpoint data."""
        self.completed = set()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


# %% Graceful Interruption Handler
class GracefulInterrupt:
    """Handle Ctrl+C gracefully, allowing current image to complete."""
    
    def __init__(self):
        self.interrupted = False
        self._original_handler = None
    
    def __enter__(self):
        self._original_handler = signal.signal(signal.SIGINT, self._handler)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._original_handler)
    
    def _handler(self, signum, frame):
        print("\n\n⚠️  Interrupt received. Finishing current image then stopping...")
        print("   (Press Ctrl+C again to force quit)\n")
        self.interrupted = True
        # Restore original handler so second Ctrl+C forces quit
        signal.signal(signal.SIGINT, self._original_handler)


# %% Helper Functions
def load_images_from_json(json_path: Path) -> List[np.ndarray]:
    """Load images from Tengyur JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = []
    for folio in data:
        # Assuming JSON structure has image data - adapt as needed
        # For now, just return empty list as placeholder
        pass
    
    return images


def get_all_test_images() -> Dict[str, List[Path]]:
    """Get all test images from JSON files."""
    images_by_file = {}
    
    # For JSON files, we'll need to extract images first
    # For now, just look for existing image files
    for json_path in sorted(TEST_SAMPLES_DIR.glob("*.json")):
        # For Tengyur JSONs, we'd need to convert to images
        # This is a placeholder - implement based on your JSON structure
        print(f"Found JSON: {json_path.name}")
        images_by_file[json_path.stem] = []
    
    # Also check for any existing image files
    for img_path in sorted(TEST_SAMPLES_DIR.glob("*.jpg")) + sorted(TEST_SAMPLES_DIR.glob("*.png")):
        file_key = img_path.stem
        if file_key not in images_by_file:
            images_by_file[file_key] = []
        images_by_file[file_key].append(img_path)
    
    return images_by_file


# %% OCR Pipeline Wrapper
class GridSearchOCR:
    """Wrapper for running OCR with different parameter combinations."""
    
    def __init__(self):
        self.platform = get_platform()
        self.ocr_models = {}
        self.pipelines = {}
        
        # Load line detection config
        self.line_config = LineDetectionConfig(
            model_file=str(LINE_MODEL_PATH),
            patch_size=512
        )
        
        # Load layout detection config
        self.layout_config = LayoutDetectionConfig(
            model_file=str(LAYOUT_MODEL_PATH),
            patch_size=512,
            classes=["background", "image", "line", "caption", "margin"]
        )
        
        # Load all OCR models
        self._load_ocr_models()
    
    def _load_ocr_models(self):
        """Load all available OCR models."""
        print("\nLoading OCR models...")
        
        models = import_local_models(str(OCR_MODELS_DIR))
        for model in models:
            self.ocr_models[model.name] = model
            print(f"  ✅ Loaded: {model.name}")
    
    def get_pipeline(self, params: GridSearchParams) -> OCRPipeline:
        """Get or create a pipeline for the given parameters."""
        # Create cache key
        cache_key = f"{params.ocr_model_name}_{params.line_mode}"
        
        if cache_key not in self.pipelines:
            ocr_model = self.ocr_models.get(params.ocr_model_name)
            if not ocr_model:
                raise ValueError(f"OCR model not found: {params.ocr_model_name}")
            
            line_config = self.line_config if params.line_mode == "line" else self.layout_config
            
            self.pipelines[cache_key] = OCRPipeline(
                platform=self.platform,
                ocr_config=ocr_model.config,
                line_config=line_config
            )
        
        return self.pipelines[cache_key]
    
    def run_ocr(self, image_path: Path, params: GridSearchParams) -> Tuple[bool, int, str, str, Dict]:
        """
        Run OCR on a single image with given parameters.
        
        Returns: (success, num_lines, ocr_text, error_message, quality_metrics)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False, 0, "", f"Failed to load image: {image_path}", {}
            
            # Get pipeline
            pipeline = self.get_pipeline(params)
            
            # Run OCR (use_tps always True, threshold controls behavior)
            status, result = pipeline.run_ocr(
                image=image,
                k_factor=params.k_factor,
                bbox_tolerance=params.bbox_tolerance,
                merge_lines=params.merge_lines,
                use_tps=True,  # Always True, threshold controls sensitivity
                tps_threshold=params.tps_threshold,
                target_encoding=Encoding.Unicode
            )
            
            if status.name == "SUCCESS":
                rot_mask, lines, ocr_lines, angle = result
                text = "\n".join([line.text for line in ocr_lines])
                
                # Score the OCR quality
                quality_metrics = quality_scorer.score_text(text)
                
                return True, len(ocr_lines), text, "", quality_metrics
            else:
                return False, 0, "", str(result), {}
                
        except Exception as e:
            return False, 0, "", f"{type(e).__name__}: {str(e)}", {}


# %% Result Saving
def save_result(
    output_dir: Path,
    file_name: str,
    image_name: str,
    params: GridSearchParams,
    success: bool,
    num_lines: int,
    ocr_text: str,
    error_message: str,
    processing_time: float,
    quality_metrics: Dict
):
    """Save a single OCR result to a text file."""
    
    # Create directory structure: output_dir/file_name/image_name/
    result_dir = output_dir / file_name / image_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename from parameters
    filename = params.to_filename() + ".txt"
    filepath = result_dir / filename
    
    # Build file content
    content = []
    content.append("=" * 70)
    content.append("OCR RESULT")
    content.append("=" * 70)
    content.append(f"")
    content.append(f"File: {file_name}")
    content.append(f"Image: {image_name}")
    content.append(f"")
    content.append("PARAMETERS:")
    content.append(f"  OCR Model: {params.ocr_model_name}")
    content.append(f"  Line Mode: {params.line_mode}")
    content.append(f"  Class Threshold: {params.class_threshold}")
    content.append(f"  K-Factor: {params.k_factor}")
    content.append(f"  BBox Tolerance: {params.bbox_tolerance}")
    content.append(f"  Merge Lines: {params.merge_lines}")
    content.append(f"  TPS Threshold: {params.tps_threshold}")
    content.append(f"")
    content.append("RESULTS:")
    content.append(f"  Success: {success}")
    content.append(f"  Lines Detected: {num_lines}")
    content.append(f"  Processing Time: {processing_time:.2f}s")
    
    # Add quality metrics
    if quality_metrics:
        content.append(f"")
        content.append("QUALITY METRICS:")
        content.append(f"  ✨ Quality Score: {quality_metrics.get('quality_score', 0):.2f}/100")
        content.append(f"  Total Tokens: {quality_metrics.get('total_tokens', 0)}")
        content.append(f"  Valid Words: {quality_metrics.get('valid_tokens', 0)}")
        content.append(f"  Invalid Words: {quality_metrics.get('invalid_tokens', 0)}")
    
    if error_message:
        content.append(f"  Error: {error_message}")
    content.append(f"")
    content.append("=" * 70)
    content.append("OCR TEXT")
    content.append("=" * 70)
    content.append(f"")
    content.append(ocr_text if ocr_text else "[No text extracted]")
    
    # Write file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))


def save_summary_csv(output_dir: Path, all_results: List[Dict]):
    """Save a summary CSV of all results for analysis."""
    
    csv_path = output_dir / "summary.csv"
    
    if not all_results:
        print("No results to save to summary")
        return
    
    # Get fieldnames from first result
    fieldnames = list(all_results[0].keys())
    
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n📊 Summary CSV saved to: {csv_path}")


# %% Main Grid Search Runner
def run_grid_search(
    max_images_per_file: int = None,
    resume: bool = True
):
    """
    Run the full grid search.
    
    Args:
        max_images_per_file: Limit images per file (for testing)
        resume: Whether to resume from checkpoint (default True)
    """
    
    # Initialize
    checkpoint = CheckpointManager(CHECKPOINT_FILE)
    ocr = GridSearchOCR()
    all_results = []  # For summary CSV
    
    if resume and checkpoint.get_completed_count() > 0:
        print(f"\n📂 Resuming from checkpoint: {checkpoint.get_completed_count()} images already completed")
    elif not resume:
        checkpoint.reset()
        print("\n🔄 Starting fresh (checkpoint cleared)")
    
    # Get test images
    print("\n" + "=" * 70)
    print("LOADING TEST IMAGES")
    print("=" * 70)
    images_by_file = get_all_test_images()
    
    # Calculate totals
    total_images = sum(len(imgs) for imgs in images_by_file.values())
    param_combinations = generate_param_combinations()
    total_iterations = total_images * len(param_combinations)
    
    print(f"\n" + "=" * 70)
    print("GRID SEARCH WITH QUALITY SCORING")
    print("=" * 70)
    print(f"Total images: {total_images}")
    print(f"Combinations per image: {len(param_combinations)}")
    print(f"Total iterations: {total_iterations}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    # Run with graceful interrupt handling
    with GracefulInterrupt() as interrupt:
        
        # Progress tracking
        images_processed = 0
        images_skipped = 0
        
        for file_name, image_paths in sorted(images_by_file.items()):
            if interrupt.interrupted:
                break
            
            print(f"\n📁 File: {file_name}")
            print(f"   Images: {len(image_paths)}")
            
            # Limit images if specified
            if max_images_per_file:
                image_paths = image_paths[:max_images_per_file]
            
            for image_path in image_paths:
                if interrupt.interrupted:
                    break
                
                image_key = str(image_path)
                image_name = image_path.stem
                
                # Skip if already completed
                if checkpoint.is_completed(image_key):
                    images_skipped += 1
                    continue
                
                print(f"\n   🖼️  Processing: {image_name}")
                
                # Progress bar for this image's combinations
                pbar = tqdm(
                    param_combinations, 
                    desc=f"      Params",
                    leave=False
                )
                
                for params in pbar:
                    start_time = time.time()
                    
                    # Run OCR
                    success, num_lines, ocr_text, error, quality_metrics = ocr.run_ocr(image_path, params)
                    
                    processing_time = time.time() - start_time
                    
                    # Save individual result file
                    save_result(
                        output_dir=OUTPUT_DIR,
                        file_name=file_name,
                        image_name=image_name,
                        params=params,
                        success=success,
                        num_lines=num_lines,
                        ocr_text=ocr_text,
                        error_message=error,
                        processing_time=processing_time,
                        quality_metrics=quality_metrics
                    )
                    
                    # Add to summary
                    all_results.append({
                        "file_name": file_name,
                        "image_name": image_name,
                        "ocr_model_name": params.ocr_model_name,
                        "line_mode": params.line_mode,
                        "class_threshold": params.class_threshold,
                        "k_factor": params.k_factor,
                        "bbox_tolerance": params.bbox_tolerance,
                        "merge_lines": params.merge_lines,
                        "tps_threshold": params.tps_threshold,
                        "success": success,
                        "num_lines_detected": num_lines,
                        "processing_time": processing_time,
                        "quality_score": quality_metrics.get('quality_score', 0.0),
                        "total_tokens": quality_metrics.get('total_tokens', 0),
                        "valid_tokens": quality_metrics.get('valid_tokens', 0),
                        "invalid_tokens": quality_metrics.get('invalid_tokens', 0),
                        "error": error[:100] if error else ""
                    })
                
                pbar.close()
                
                # Mark image as completed
                checkpoint.mark_completed(image_key)
                images_processed += 1
                print(f"      ✅ Completed ({images_processed} processed, {images_skipped} skipped)")
        
        # Save summary CSV
        save_summary_csv(OUTPUT_DIR, all_results)
    
    # Final status
    print("\n" + "=" * 70)
    if interrupt.interrupted:
        print("⚠️  INTERRUPTED - Progress saved, run again to resume")
    else:
        print("✅ GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Images processed this run: {images_processed}")
    print(f"Images skipped (from checkpoint): {images_skipped}")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    return all_results


# %% Analysis Helper
def analyze_results():
    """Load and analyze results from the summary CSV with quality scores."""
    
    csv_path = OUTPUT_DIR / "summary.csv"
    
    if not csv_path.exists():
        print("❌ No summary.csv found. Run grid search first.")
        return None
    
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas required for analysis. Install with: pip install pandas")
        return None
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS WITH QUALITY SCORES")
    print("=" * 70)
    
    print(f"\nTotal results: {len(df)}")
    
    print("\n📊 Success Rate by OCR Model:")
    print(df.groupby('ocr_model_name')['success'].mean().sort_values(ascending=False))
    
    print("\n✨ Average Quality Score by OCR Model:")
    successful = df[df['success'] == True]
    if len(successful) > 0:
        print(successful.groupby('ocr_model_name')['quality_score'].mean().sort_values(ascending=False))
    
    print("\n📊 Success Rate by Line Mode:")
    print(df.groupby('line_mode')['success'].mean())
    
    print("\n✨ Average Quality Score by Line Mode:")
    if len(successful) > 0:
        print(successful.groupby('line_mode')['quality_score'].mean())
    
    print("\n📊 Success Rate by K-Factor:")
    print(df.groupby('k_factor')['success'].mean())
    
    print("\n✨ Average Quality Score by K-Factor:")
    if len(successful) > 0:
        print(successful.groupby('k_factor')['quality_score'].mean())
    
    # Top 10 parameter combinations by quality score
    print("\n🏆 TOP 10 PARAMETER COMBINATIONS (by quality score):")
    if len(successful) > 0:
        top_10 = successful.nlargest(10, 'quality_score')[
            ['file_name', 'image_name', 'ocr_model_name', 'line_mode', 
             'k_factor', 'bbox_tolerance', 'quality_score', 'num_lines_detected']
        ]
        print(top_10.to_string(index=False))
    
    # Quality score distribution
    print("\n📊 Quality Score Distribution:")
    print(f"  >90 (Excellent):  {len(df[df['quality_score'] > 90])}")
    print(f"  70-90 (Good):     {len(df[(df['quality_score'] >= 70) & (df['quality_score'] <= 90)])}")
    print(f"  50-70 (Fair):     {len(df[(df['quality_score'] >= 50) & (df['quality_score'] < 70)])}")
    print(f"  <50 (Poor):       {len(df[df['quality_score'] < 50])}")
    
    return df


# %% Quick Test
def quick_test():
    """Run a quick test with 1 image, limited params."""
    print("\n" + "=" * 70)
    print("QUICK TEST")
    print("=" * 70)
    print("Running with 1 image to verify setup...")
    
    # Clear checkpoint for fresh test
    checkpoint = CheckpointManager(CHECKPOINT_FILE)
    checkpoint.reset()
    
    results = run_grid_search(
        max_images_per_file=1,
        resume=False
    )
    
    return results


# %% 1. QUICK TEST - Run this first to verify setup
"""
Runs 1 image with all parameter combinations.
Use this to verify everything works before the full run.
"""
# quick_test_results = quick_test()

# %% 2. FULL GRID SEARCH - Main run (resume-able)
"""
Runs all images with all parameter combinations.
- Automatically resumes from checkpoint if interrupted
- Press Ctrl+C to stop gracefully (finishes current image)
- Run this cell again to continue where you left off
"""
# full_results = run_grid_search()

# %% 3. ANALYZE RESULTS - View summary statistics
"""
Loads the summary.csv and shows statistics including quality scores.
Run this after the grid search completes (or partially completes).
"""
# df = analyze_results()