# EEG-JEPA

## Overview

This repository contains the code and experiments for EEG-JEPA, the first application of Joint-Embedding Predictive Architecture (JEPA) for electroencephalogram (EEG) based user identification. Please see the full thesis paper for further context. 

**Thesis:** [Overleaf](https://www.overleaf.com/read/wmxqybszcnwr#ec719a) | [PDF](https://drive.google.com/file/d/1YveDL35y8MgITF9k9JbsLWuFjHvHBRlz/view?usp=sharing)

**Code:** [GitHub Repository](https://github.com/belindahu02/EEG-JEPA)

## Repository Structure

### 1. Data Pre-processing ([data-pipeline/](https://github.com/belindahu02/EEG-JEPA/tree/main/data-pipeline))

Contains tools for converting and processing EEG data:
- EDF (MMI) and CSV (MusicID) to spectrogram converters
- Spectrogram to embedding using JEPA encoder (`precompute.py`)
- Embedding regrouping (`group.py`) for classification
- Augmentations implemented based on [BreathPrint paper, Section 3.1](https://dl.acm.org/doi/pdf/10.1145/3287036)

**Note:** Files with suffix `_new` have new augmentations and more aggressive windowing to increase MusicID data volume further. These have yet to be fully tested.

### 2. JEPA Training ([JEPA/](https://github.com/belindahu02/EEG-JEPA/tree/main/JEPA))

Implementation largely based on Sony's Audio-JEPA ([repository](https://github.com/SonyCSLParis/audio-representations/tree/master/src/models), [paper](https://arxiv.org/abs/2405.08679)).

**Requirements:**
- Install dependencies from [`requirements.txt`](https://github.com/belindahu02/EEG-JEPA/blob/main/requirements.txt)
- Data inputs must be in **spectrogram** format, use converters detailed above
- Relative file paths should be listed in `data/files_audioset.csv` ([examples](https://github.com/belindahu02/EEG-JEPA/tree/main/data))

**Usage:**
```bash
dora run
```

- Training artifacts will be saved in `logs/`

### 3. Scaling Experiment ([experiments/scaling/](https://github.com/belindahu02/EEG-JEPA/tree/main/experiments/scaling))

- Install dependencies from [`experiments/requirements.txt`](https://github.com/belindahu02/EEG-JEPA/blob/main/experiments/requirements.txt)
- Generate embeddings using JEPA encoder checkpoint with `precompute.py` based on a CSV list (same format as `data/files_audioset.csv`) and group channels using `group.py`
- Entry point: `plot_results.py` for each dataset and baseline
- **Functional baselines:** supervised, data augmentation, and multi-task
- **Note:** [SimSiam](https://github.com/belindahu02/EEG-JEPA/tree/main/experiments/scaling/baselines/mmi/simsiam) does not successfully recreate the accuracy shown in [Oshan's paper](https://arxiv.org/abs/2210.12964)
- All ablation and overfitting experiments are in [experiments/scaling/ablation/](https://github.com/belindahu02/EEG-JEPA/tree/main/experiments/scaling/ablation) 

### 4. Masking Experiment ([experiments/masking/](https://github.com/belindahu02/EEG-JEPA/tree/main/experiments/masking))

- Install dependencies from [`experiments/requirements.txt`](https://github.com/belindahu02/EEG-JEPA/blob/main/experiments/requirements.txt)
- Generate embeddings using JEPA encoder checkpoint with `precompute.py` based on a CSV list (same format as `data/files_audioset.csv`) and group channels using `group.py`
- Entry point: `experiment.py` for each dataset and baseline
- **Functional baselines:** supervised and data augmentation

### 5. Other Code

Outdated initial implementations kept for reference
- [classification/](https://github.com/belindahu02/EEG-JEPA/tree/main/classification): Contains the initial classifier implementations, including for 1D with dimensionality reduction and 2D for direct spectrograms (which was ultimately adopted)
- [baselines/](https://github.com/belindahu02/EEG-JEPA/tree/main/baselines): Contains initial recreation of baselines from [Oshan's paper](https://arxiv.org/abs/2210.12964)

## Datasets

- [EEG Motor Movement/Imagery Dataset (MMI)](https://physionet.org/content/eegmmidb/1.0.0/)
- [MusicID Dataset](https://www.kaggle.com/datasets/570da2409d68350abacc3d1119faabe0a79bf45a3ae5c32a38ab007d8326da1a)

## Key References

- EEG-MMI 
- [Investigating design choices in joint-embedding predictive architectures for general audio representation learning (Riou et al.)](https://arxiv.org/abs/2405.08679)
- [Non-Contrastive Learning-based Behavioural Biometrics for Smart IoT Devices (Jayawardana et al.)](https://arxiv.org/abs/2210.12964)
- [Performance Characterization of Deep Learning Models for Breathing-based Authentication on Resource-Constrained Devices (Chauhan et al.)](https://dl.acm.org/doi/pdf/10.1145/3287036)