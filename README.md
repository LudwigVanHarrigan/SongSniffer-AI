# SongSniffer-AI
A machine learning project for distinguishing between AI-generated and human-created music using convolutional neural networks (CNNs) trained on mel-spectrograms.

This was developed by Kevin Cotellesso, Alex Stedman, Reese Sieger, and Jacob Richardson for MA416: Deep Learning at Rose-Hulman Institute of Technology. 

**Disclaimer: this readme was written primarily by Claude Sonnet 4, but was proofread and edited by Kevin Cotellesso.**


## 📁 Repository Structure

### **Dataset Generation Scripts** (`dataset_gen_scripts/`)
- `make_dataset_V2.py` - Main dataset creation script with train/test split functionality. Used to make SmellySongs9k.
- `make_dataset.py` - Original dataset creation script
- `extract_mel_spectrograms.py` - Generates mel-spectrograms from audio files found in a folder

### **Datasets** (`Datasets/`)
**ON RHIT CSSE SERVERS ONLY** Contains various dataset versions and source data:
- `SmellySongs9k_V2/` - Same as SS9k but all files are at 44100hz sample rate (THANK YOU ffmpeg)
- `SmellySongs9k/` - DEPRECATED - 9,000 sample dataset with proper train/test split, balancing and greyscale spectrograms.
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
  - `RFCtry2.ipynb` - Alex's second attempt at random forest classifier, run on SmellySongs9k_V2
  - `Webpage.py` - A script that runs a flask server to host an online demo of SimpleCNN1

### **Development and Prototype Implementations**
- `CNN_prototypes/` - CNN development notebooks
- `classifier/` - Basic classifier implementations


## 👃SmellySongs9k (V2) Structure

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

## 📊 SimpleCNN1 Versions
- T1: The original training. 20 epochs on SmellySongs9k
- T2: A lighter training on SmellySongs9k
- T3: 20 (?) epochs on SmellySongs9k_V2. No longer sensitive to sampling rate.
- T4: A center-cropped model trained 3 epochs on SmellySongs9k. (BEST)
- T5: A center-cropped model trained on 20 epochs on SmellySongs9k


## SourceFilePenTesting

A fun set of samples that may or may not break our model. Have fun!!