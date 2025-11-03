#!/usr/bin/env python3
"""
Create a SmellySongs dataset folder for a binary audio ML dataset with train/test split.

V2: 
 - Train/test split functionality (80% train, 20% test).
 - Required conversion to .wav and splitting to specified seconds.
 - Organizes files into AI/Human folders instead of CSV format.

What it does:
- Scans two source folders for audio files:
    * Datasets/RoyaltyFree/Audio/ -> Human audio
    * Datasets/SunoCaps/audio/audio/ -> AI audio
- Splits each source into 80% train and 20% test
- Converts all files to .wav and splits them into specified-length chunks
- Organizes files into folder structure:
    * out_dir/train/AI/
    * out_dir/train/Human/
    * out_dir/test/AI/
    * out_dir/test/Human/
- Optionally generates mel spectrograms for each audio chunk:
    * out_dir/train/AI/spectrograms/
    * out_dir/train/Human/spectrograms/
    * out_dir/test/AI/spectrograms/
    * out_dir/test/Human/spectrograms/
    * Each spectrogram directory includes metadata.json with processing details

Usage examples:
    # Create dataset with 5-second chunks
    python make_dataset_V2.py --out-dir Datasets/MLDataset --split-seconds 5

    # Create dataset with spectrograms
    python make_dataset_V2.py --out-dir Datasets/MLDataset --split-seconds 5 --generate-spectrograms

    # Create dataset with custom spectrogram settings
    python make_dataset_V2.py \
            --out-dir Datasets/MLDataset \
            --split-seconds 10 \
            --generate-spectrograms \
            --spectrogram-resolution 256 \
            --spectrogram-colormap plasma

    # Override input folders explicitly
    python make_dataset_V2.py \
            --real Datasets/RoyaltyFree/Audio \
            --ai Datasets/SunoCaps/audio/audio \
            --out-dir Datasets/MLDataset \
            --split-seconds 10
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".wma", ".aiff", ".aif", ".aifc"
}


def list_audio_files(directory: Path) -> List[Path]:
    """Recursively list audio file paths in a directory.

    Returns full Paths for each audio file found.
    Non-audio files are ignored. Missing directories yield an empty list.
    """
    if not directory.exists():
        return []
    files: List[Path] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


essential_description = (
    "Creates train/test split dataset with AI and Human folders from two source folders.\n"
    "Converts all files to .wav and splits them into specified-length chunks."
)


def split_files_train_test(files: List[Path], test_ratio: float = 0.2) -> Tuple[List[Path], List[Path]]:
    """Split a list of files into train and test sets."""
    shuffled = files.copy()
    random.shuffle(shuffled)
    test_count = int(len(shuffled) * test_ratio)
    return shuffled[test_count:], shuffled[:test_count]


def generate_mel_spectrogram(audio_path: Path, output_path: Path, resolution: int = 224, 
                           colormap: str = 'viridis', grayscale: bool = False) -> Tuple[float, np.ndarray]:
    """
    Generate a square Mel spectrogram image from an audio file.
    
    Args:
        audio_path: Path to input .wav file
        output_path: Path to save output .png file
        resolution: Image resolution (width and height in pixels)
        colormap: Matplotlib colormap to use
        grayscale: If True, save as grayscale; otherwise use colormap
    
    Returns:
        tuple: (audio_duration, pixel_values_array)
    """
    try:
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from PIL import Image
    except ImportError as e:
        print(
            "Error: Mel spectrogram generation requires additional packages.\n"
            "Install with: pip install librosa matplotlib pillow\n"
        )
        raise SystemExit(1) from e
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=resolution)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure without axes/borders
    dpi = 100
    fig_size = resolution / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = Axes(fig, (0., 0., 1., 1.))
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Plot spectrogram
    if grayscale:
        ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='gray')
    else:
        ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap=colormap)
    
    # Save image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    
    # Load saved image to get actual pixel values
    img = Image.open(output_path)
    pixel_values = np.array(img)
    
    # Calculate audio duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration, pixel_values


def process_and_copy_files(
    real_dir: Path,
    ai_dir: Path,
    out_dir: Path,
    split_seconds: float,
    generate_spectrograms: bool = False,
    spectrogram_resolution: int = 224,
    spectrogram_grayscale: bool = False,
    spectrogram_colormap: str = 'viridis',
) -> Tuple[int, int, int, int]:
    """Process audio files and organize them into train/test AI/Human folder structure.
    
    Args:
        real_dir: Directory containing human audio files
        ai_dir: Directory containing AI audio files  
        out_dir: Output directory for processed files
        split_seconds: Length of audio chunks in seconds
        generate_spectrograms: Whether to generate mel spectrograms
        spectrogram_resolution: Resolution for spectrogram images
        spectrogram_grayscale: Whether to generate grayscale spectrograms
        spectrogram_colormap: Colormap for color spectrograms
    
    Returns: (train_human_count, train_ai_count, test_human_count, test_ai_count)
    """
    # Get all files from source directories
    real_files = sorted(list_audio_files(real_dir))
    ai_files = sorted(list_audio_files(ai_dir))

    if len(real_files) == 0 and len(ai_files) == 0:
        return 0, 0, 0, 0

    # Ensure pydub is available
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception as e:
        print(
            "Error: audio operations require the 'pydub' package.\n"
            "Install with: pip install pydub\n"
            "Note: pydub also needs ffmpeg installed on your system."
        )
        raise SystemExit(1) from e

    # Validate split_seconds
    try:
        split_ms = int(float(split_seconds) * 1000)
    except Exception:
        print("Error: --split-seconds must be a number (seconds)")
        raise SystemExit(2)
    if split_ms <= 0:
        print("Error: --split-seconds must be > 0")
        raise SystemExit(2)

    # Split files into train/test
    real_train, real_test = split_files_train_test(real_files)
    ai_train, ai_test = split_files_train_test(ai_files)

    # Create output directories
    train_human_dir = out_dir / "train" / "Human"
    train_ai_dir = out_dir / "train" / "AI"
    test_human_dir = out_dir / "test" / "Human"
    test_ai_dir = out_dir / "test" / "AI"

    # Create spectrogram directories if needed
    spec_dirs = []
    if generate_spectrograms:
        train_human_spec_dir = out_dir / "train" / "Human" / "spectrograms"
        train_ai_spec_dir = out_dir / "train" / "AI" / "spectrograms"
        test_human_spec_dir = out_dir / "test" / "Human" / "spectrograms"
        test_ai_spec_dir = out_dir / "test" / "AI" / "spectrograms"
        spec_dirs = [train_human_spec_dir, train_ai_spec_dir, test_human_spec_dir, test_ai_spec_dir]

    for dir_path in [train_human_dir, train_ai_dir, test_human_dir, test_ai_dir] + spec_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Process files and keep counts
    counts = [0, 0, 0, 0]  # train_human, train_ai, test_human, test_ai
    file_counters = [1, 1, 1, 1]  # separate counters for each directory
    
    # Metadata for spectrograms
    spectrogram_metadata = {
        'resolution': spectrogram_resolution,
        'grayscale': spectrogram_grayscale,
        'colormap': None if spectrogram_grayscale else spectrogram_colormap,
        'categories': {
            'train_human': {'files': []},
            'train_ai': {'files': []},
            'test_human': {'files': []},
            'test_ai': {'files': []}
        }
    } if generate_spectrograms else None

    # Process each category
    category_names = ['train_human', 'train_ai', 'test_human', 'test_ai']
    categories = [
        (real_train, train_human_dir, 0, category_names[0]),
        (ai_train, train_ai_dir, 1, category_names[1]), 
        (real_test, test_human_dir, 2, category_names[2]),
        (ai_test, test_ai_dir, 3, category_names[3])
    ]

    for files, target_dir, count_idx, category_name in categories:
        spec_dir = None
        if generate_spectrograms:
            spec_dir = target_dir / "spectrograms"
            
        for src in files:
            try:
                # Load audio and split into chunks
                audio = AudioSegment.from_file(src)
                
                # Process each chunk
                for start in range(0, len(audio), split_ms):
                    end = min(start + split_ms, len(audio))
                    segment = audio[start:end]
                    
                    # Skip segments shorter than requested length
                    if len(segment) < split_ms:
                        continue
                    
                    # Generate filename and save audio
                    filename = f"{file_counters[count_idx]:06d}.wav"
                    dst = target_dir / filename
                    segment.export(dst, format="wav")
                    
                    # Generate spectrogram if requested
                    if generate_spectrograms and spec_dir:
                        try:
                            spec_filename = f"{file_counters[count_idx]:06d}.png"
                            spec_dst = spec_dir / spec_filename
                            duration, pixel_values = generate_mel_spectrogram(
                                dst, spec_dst, spectrogram_resolution, 
                                spectrogram_colormap, spectrogram_grayscale
                            )
                            
                            # Store spectrogram metadata
                            if spectrogram_metadata is not None:
                                file_meta = {
                                    'audio_file': filename,
                                    'spectrogram_file': spec_filename,
                                    'duration_seconds': float(duration),
                                    'image_shape': list(pixel_values.shape)
                                }
                                spectrogram_metadata['categories'][category_name]['files'].append(file_meta)
                            
                        except Exception as e:
                            print(f"Warning: failed to generate spectrogram for {filename} -> {e}")
                    
                    file_counters[count_idx] += 1
                    counts[count_idx] += 1
                    
            except Exception as e:
                print(f"Warning: failed to process {src} -> {e}")
                continue

    # Save spectrogram metadata if generated
    if generate_spectrograms and spectrogram_metadata is not None:
        for category_name in category_names:
            category_dir = out_dir / category_name.replace('_', '/') / "spectrograms"
            if category_dir.exists():
                metadata_path = category_dir / "metadata.json"
                category_data = {
                    'resolution': spectrogram_metadata['resolution'],
                    'grayscale': spectrogram_metadata['grayscale'],
                    'colormap': spectrogram_metadata['colormap'],
                    'files': spectrogram_metadata['categories'][category_name]['files']
                }
                with open(metadata_path, 'w') as f:
                    json.dump(category_data, f, indent=2)
    
    return counts[0], counts[1], counts[2], counts[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=essential_description)

    # Default roots relative to repository root (this script is in scripts/)
    repo_root = Path(__file__).resolve().parents[1]
    default_real = str(repo_root / "Datasets" / "RoyaltyFree" / "Audio")
    default_ai = str(repo_root / "Datasets" / "SunoCaps" / "audio" / "audio")

    parser.add_argument(
        "--real",
        type=str,
        default=default_real,
        help=f"Path to Human audio folder (default: {default_real})",
    )
    parser.add_argument(
        "--ai",
        type=str,
        default=default_ai,
        help=f"Path to AI audio folder (default: {default_ai})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output dataset directory to create train/test/AI/Human folder structure",
    )
    parser.add_argument(
        "--split-seconds",
        type=float,
        required=True,
        help="Split each file into fixed-length chunks (in seconds). Required parameter.",
    )
    parser.add_argument(
        "--generate-spectrograms",
        action="store_true",
        help="Generate mel spectrograms for each audio file (requires librosa, matplotlib, pillow)",
    )
    parser.add_argument(
        "--spectrogram-resolution",
        type=int,
        default=224,
        help="Resolution for spectrogram images (width and height in pixels, default: 224)",
    )
    parser.add_argument(
        "--spectrogram-grayscale",
        action="store_true",
        help="Generate grayscale spectrograms instead of color",
    )
    parser.add_argument(
        "--spectrogram-colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for color spectrograms (default: viridis)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_dir = Path(args.real)
    ai_dir = Path(args.ai)
    out_dir = Path(args.out_dir)

    print(f"Human dir: {real_dir}")
    print(f"AI dir:    {ai_dir}")
    print(f"Output dataset:  {out_dir}")
    print(f"Split seconds: {args.split_seconds}")
    if args.generate_spectrograms:
        print(f"Generating {'grayscale' if args.spectrogram_grayscale else 'color'} spectrograms at {args.spectrogram_resolution}x{args.spectrogram_resolution}")
        if not args.spectrogram_grayscale:
            print(f"Colormap: {args.spectrogram_colormap}")

    # Set random seed for reproducible train/test splits
    random.seed(42)

    train_human, train_ai, test_human, test_ai = process_and_copy_files(
        real_dir,
        ai_dir,
        out_dir,
        args.split_seconds,
        args.generate_spectrograms,
        args.spectrogram_resolution,
        args.spectrogram_grayscale,
        args.spectrogram_colormap,
    )
    
    total_files = train_human + train_ai + test_human + test_ai
    print(f"\nProcessed {total_files} audio chunks:")
    print(f"  Train: {train_human + train_ai} files (Human: {train_human}, AI: {train_ai})")
    print(f"  Test:  {test_human + test_ai} files (Human: {test_human}, AI: {test_ai})")

    if total_files == 0:
        print("\nNo audio files found with known extensions.")
    else:
        print(f"\nDataset structure created in {out_dir}:")
        print("  train/Human/")
        print("  train/AI/")
        print("  test/Human/") 
        print("  test/AI/")
        if args.generate_spectrograms:
            print("\nSpectrogram directories:")
            print("  train/Human/spectrograms/")
            print("  train/AI/spectrograms/")
            print("  test/Human/spectrograms/")
            print("  test/AI/spectrograms/")
            print("  (Each with metadata.json containing spectrogram details)")


if __name__ == "__main__":
    main()
