import os
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import numpy as np

def generate_mel_spectrogram(audio_path, output_path, resolution=224):
    """
    Generate a square Mel spectrogram image from an audio file.

    Args:
        audio_path: Path to input .wav file
        output_path: Path to save output .png file
        resolution: Image resolution (width and height in pixels)

    Returns:
        None
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
    ax.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='gray')

    # Save image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)


def process_wav_to_spectrogram(input_dir, output_dir, resolution=224):
    """
    Process all .wav files in the input directory and save their spectrograms to the output directory.

    Args:
        input_dir: Directory containing .wav files
        output_dir: Directory to save spectrogram images
        resolution: Image resolution (width and height in pixels)

    Returns:
        None
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .wav files
    wav_files = sorted(list(input_path.glob('*.wav')))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} .wav files")
    print(f"Generating grayscale mel spectrograms of resolution {resolution}x{resolution}px")

    for wav_file in wav_files:
        # Generate output filename
        image_filename = wav_file.stem + '.png'
        output_file = output_path / image_filename
        generate_mel_spectrogram(wav_file, output_file, resolution)

if __name__ == "__main__":
    input_directory = "Samples/"
    output_directory = "Spectrograms/"
    process_wav_to_spectrogram(input_directory, output_directory)