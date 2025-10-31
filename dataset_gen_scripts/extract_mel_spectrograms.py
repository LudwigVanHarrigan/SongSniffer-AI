#!/usr/bin/env python3
"""
Extract Mel spectrograms from .wav files in a folder.
Generates square spectrogram images in PNG format with specified resolution.
Optionally updates dataset.csv and generates metadata JSON.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def generate_mel_spectrogram(audio_path, output_path, resolution=224, colormap='viridis', grayscale=False):
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
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=resolution)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure without axes/borders
    dpi = 100
    fig_size = resolution / dpi
    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
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
    from PIL import Image
    img = Image.open(output_path)
    pixel_values = np.array(img)
    
    # Calculate audio duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    return duration, pixel_values


def process_folder(input_folder, output_folder=None, resolution=224, grayscale=False, colormap='viridis'):
    """
    Process all .wav files in a folder and generate Mel spectrograms.
    
    Args:
        input_folder: Path to folder containing .wav files
        output_folder: Path to save spectrogram images (default: input_folder/spectrograms)
        resolution: Image resolution in pixels
        grayscale: If True, generate grayscale images
        colormap: Matplotlib colormap for color images
    """
    input_path = Path(input_folder)
    
    # Set output folder
    if output_folder is None:
        output_path = input_path / 'spectrograms'
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files
    wav_files = sorted(list(input_path.glob('*.wav')))
    
    if not wav_files:
        print(f"No .wav files found in {input_folder}")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    print(f"Generating {'grayscale' if grayscale else 'color'} spectrograms at {resolution}x{resolution} resolution")
    
    # Process each file
    metadata = {
        'resolution': resolution,
        'grayscale': grayscale,
        'colormap': None if grayscale else colormap,
        'files': []
    }
    
    all_pixel_values = []
    audio_filenames = []
    image_filenames = []
    
    for wav_file in tqdm(wav_files, desc="Processing"):
        # Generate output filename
        image_filename = wav_file.stem + '.png'
        output_file = output_path / image_filename
        
        try:
            # Generate spectrogram
            duration, pixel_values = generate_mel_spectrogram(
                wav_file, output_file, resolution, colormap, grayscale
            )
            
            # Store metadata
            file_meta = {
                'audio_file': wav_file.name,
                'image_file': image_filename,
                'duration_seconds': float(duration),
                'image_shape': list(pixel_values.shape)
            }
            metadata['files'].append(file_meta)
            
            # Collect pixel values for statistics
            all_pixel_values.append(pixel_values.flatten())
            audio_filenames.append(wav_file.name)
            image_filenames.append(image_filename)
            
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
    
    # Calculate overall statistics
    if all_pixel_values:
        all_pixels = np.concatenate(all_pixel_values)
        metadata['pixel_statistics'] = {
            'mean': float(np.mean(all_pixels)),
            'std': float(np.std(all_pixels)),
            'min': float(np.min(all_pixels)),
            'max': float(np.max(all_pixels))
        }
        
        # Calculate average audio duration
        durations = [f['duration_seconds'] for f in metadata['files']]
        metadata['average_audio_duration'] = float(np.mean(durations))
    
    # Save metadata JSON
    json_path = output_path / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGenerated {len(metadata['files'])} spectrograms")
    print(f"Output directory: {output_path}")
    print(f"Metadata saved to: {json_path}")
    
    # Update dataset.csv if it exists
    csv_path = input_path / 'dataset.csv'
    if csv_path.exists():
        print(f"\nUpdating {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Create a mapping from audio filename to image filename
        filename_map = dict(zip(audio_filenames, image_filenames))
        
        # Add image_filename column if it doesn't exist
        if 'image_filename' not in df.columns:
            df['image_filename'] = df['filename'].map(filename_map) if 'filename' in df.columns else None
        else:
            # Update existing column
            if 'filename' in df.columns:
                df['image_filename'] = df['filename'].map(filename_map)
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"Added/updated 'image_filename' column in dataset.csv")
    
    print("\nPixel statistics:")
    if 'pixel_statistics' in metadata:
        stats = metadata['pixel_statistics']
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std:  {stats['std']:.2f}")
        print(f"  Min:  {stats['min']:.2f}")
        print(f"  Max:  {stats['max']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Mel spectrograms from .wav files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_folder', type=str,
                        help='Path to folder containing .wav files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder for spectrograms (default: input_folder/spectrograms)')
    parser.add_argument('-r', '--resolution', type=int, default=224,
                        help='Image resolution (width and height in pixels)')
    parser.add_argument('-g', '--grayscale', action='store_true',
                        help='Generate grayscale spectrograms instead of color')
    parser.add_argument('-c', '--colormap', type=str, default='viridis',
                        help='Matplotlib colormap for color spectrograms')
    
    args = parser.parse_args()
    
    process_folder(
        args.input_folder,
        args.output,
        args.resolution,
        args.grayscale,
        args.colormap
    )


if __name__ == '__main__':
    main()
