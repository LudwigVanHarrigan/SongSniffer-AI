# SongSniffer-AI
A machine learning project for distinguishing between AI-generated and human-created music using convolutional neural networks (CNNs) trained on mel-spectrograms.

This was developed by Kevin Cotellesso, Alex Stedman, Reese Sieger, and Jacob Richardson for MA416: Deep Learning at Rose-Hulman Institute of Technology. 

**Disclaimer: this readme was written primarily by Claude Sonnet 4, but was proofread and edited by Kevin Cotellesso.**


## 📁 Repository Structure

### **Dataset Generation Scripts** (`dataset_gen_scripts/`)
- `make_dataset_V2.py` - Main dataset creation script with train/test split functionality
- `make_dataset.py` - Original dataset creation script
- `extract_mel_spectrograms.py` - Generates mel-spectrograms from audio files

### **Datasets** (`Datasets/`)
Contains various dataset versions and source data:
- `SmellySongs9k/` - 9,000 sample dataset with proper train/test split, balancing and greyscale spectrograms.
- `SmellySongs23K/` - DEPRECATED - 23,000 sample dataset with grayscale spectrograms.
- `SmellySongs772/` - DEPRECATED - Smaller 772 sample dataset (not split into 5s chunks or spectrograms)
- `Source_Datasets/` - Raw source datasets used to generate our AI vs Human datasets
  - `RoyaltyFree/` - Human-created music
  - `SunoCaps/` - AI-generated music

### **Final Models & Training**
- `SimpleCNN1/` - Our first working model!!
  - `SimpleCNN1.py` - Model definition
  - `SimpleCNN_TRAIN.ipynb` - Training notebook
  - `SimpleCNN_inference.ipynb` - Inference notebook

### **Development and Prototype Implementations**
- `CNN_prototypes/` - CNN development notebooks
- `classifier/` - Basic classifier implementations


## 📊 SmellySongs9k Structure

The processed SmellySongs9k dataset follows this structure:
```
dataset/
├── train/
│   ├── AI/          # AI-generated music spectrograms
│   └── Human/       # Human-created music spectrograms
└── test/
    ├── AI/
    └── Human/
```
