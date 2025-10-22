#!/usr/bin/env python3
"""
Create a holistic dataset folder for a binary audio ML dataset.

What it does:
- Scans two source folders for audio files:
    * Datasets/RoyaltyFree/Audio/ -> is_AI = 0
    * Datasets/SunoCaps/audio/audio/ -> is_AI = 1
- Copies all found audio files into a specified dataset output folder, renaming
    each to an incrementing filename like 000001.wav, 000002.mp3, ...
- Produces a CSV INSIDE the dataset folder with columns: [filename, is_AI]
    where `filename` is the NEW incrementing name.
 - Optional: --convert-to-wav converts all files to .wav while copying (uses pydub)

Usage examples:
    # Use defaults for input folders and write dataset to Datasets/MLDataset
    python scripts/make_dataset_dataframe.py --out-dir Datasets/MLDataset

    # Override input folders explicitly
    python scripts/make_dataset_dataframe.py \
            --real Datasets/RoyaltyFree/Audio \
            --ai Datasets/SunoCaps/audio/audio \
            --out-dir Datasets/MLDataset

    # Convert to WAV on the fly
    python scripts/make_dataset_dataframe.py --out-dir Datasets/MLDataset --convert-to-wav
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd

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
    "Creates a DataFrame with columns [filename, is_AI] from two source folders.\n"
    "Real files are labeled 0; AI files are labeled 1."
)


def copy_files_and_build_dataframe(
    real_dir: Path, ai_dir: Path, out_dir: Path, *, convert_to_wav: bool = False
) -> pd.DataFrame:
    """Copy audio into out_dir with incrementing names and return label DataFrame.

    - Real files are labeled 0; AI files are labeled 1.
    - Filenames are sequential with zero padding based on total file count.
    - CSV creation happens in the caller.
    """
    real_files = sorted(list_audio_files(real_dir))
    ai_files = sorted(list_audio_files(ai_dir))

    pairs: List[tuple[Path, int]] = [(p, 0) for p in real_files] + [(p, 1) for p in ai_files]

    total = len(pairs)
    if total == 0:
        return pd.DataFrame([], columns=["filename", "is_AI"])  # empty

    out_dir.mkdir(parents=True, exist_ok=True)

    # If converting, ensure pydub is available early
    if convert_to_wav:
        try:
            from pydub import AudioSegment  # type: ignore
        except Exception as e:
            print(
                "Error: --convert-to-wav requires the 'pydub' package.\n"
                "Install with: pip install pydub\n"
                "and"
                "pip install audioop-lts\n"
                "Note: pydub also needs ffmpeg installed on your system."
            )
            raise SystemExit(1) from e

    width = max(6, len(str(total)))  # e.g., 000001
    rows: List[dict] = []
    idx = 0
    for src, label in pairs:
        try:
            ext = ".wav" if convert_to_wav else src.suffix
            candidate_idx = idx + 1
            new_name = f"{candidate_idx:0{width}d}{ext}"
            dst = out_dir / new_name

            if convert_to_wav and src.suffix != ".wav":
                from pydub import AudioSegment  # type: ignore
                audio = AudioSegment.from_file(src)
                audio.export(dst, format="wav")
                print("Converted:", src, "->", dst)
            else:
                shutil.copy2(src, dst)

            idx += 1
            rows.append({"filename": new_name, "is_AI": label})
        except Exception as e:
            print(f"Warning: failed to process {src} -> {e}")
            continue

    df = pd.DataFrame(rows, columns=["filename", "is_AI"])  # enforce column order
    return df


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
        help=f"Path to Real audio folder (default: {default_real})",
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
        help="Output dataset directory to copy files into and write dataset.csv",
    )
    parser.add_argument(
        "--convert-to-wav",
        action="store_true",
        help="Convert all audio to .wav while copying (requires pydub and ffmpeg)",
    )
    # Deprecated: keep for compatibility but ignored when out-dir is used
    parser.add_argument(
        "--to-csv",
        type=str,
        default=None,
        help="Deprecated. CSV is always written to <out-dir>/dataset.csv",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    real_dir = Path(args.real)
    ai_dir = Path(args.ai)
    out_dir = Path(args.out_dir)

    print(f"Real dir: {real_dir}")
    print(f"AI dir:    {ai_dir}")
    print(f"Output dataset:  {out_dir}")

    df = copy_files_and_build_dataframe(
        real_dir, ai_dir, out_dir, convert_to_wav=args.convert_to_wav
    )
    print(
        f"Copied {len(df)} files (Real={ (df['is_AI']==0).sum() }, AI={ (df['is_AI']==1).sum() })"
    )

    # Preview a few rows
    if not df.empty:
        print("\nSample:")
        print(df.head(10).to_string(index=False))
    else:
        print("\nNo audio files found with known extensions.")

    # Always write CSV to the output dataset directory
    csv_path = out_dir / "dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote CSV -> {csv_path}")


if __name__ == "__main__":
    main()
