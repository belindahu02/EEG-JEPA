"""
Embedding extraction for masking experiment using pre-trained JEPA encoder
Based on your original precompute.py - uses Hydra config for model instantiation
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hydra
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F


class MaskingEvalDataset(Dataset):
    """
    Dataset for embedding extraction in masking experiment
    Based on your original EvalDataset - EXACTLY the same
    """
    def __init__(self, csv_file, data_dir, crop_frames=208, repeat_short=True):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.crop_frames = crop_frames
        self.repeat_short = repeat_short

        print(f"MaskingEvalDataset created with {len(self.df)} files")

    def __len__(self):
        return len(self.df)

    def complete_audio(self, lms):
        """Complete audio processing - EXACTLY from your original"""
        l = lms.shape[-1]
        # repeat if shorter than crop_frames
        if self.repeat_short and l < self.crop_frames:
            while l < self.crop_frames:
                lms = torch.cat([lms, lms], dim=-1)
                l = lms.shape[-1]

        # crop if longer than crop_frames
        if l > self.crop_frames:
            lms = lms[..., :self.crop_frames]  # take first crop_frames frames
        # pad if shorter
        elif l < self.crop_frames:
            pad_param = [0, self.crop_frames - l] + [0, 0] * (lms.ndim - 1)
            lms = F.pad(lms, pad_param, mode='constant', value=0)

        return lms

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        data = np.load(file_path)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]  # add channel dim
        data = torch.tensor(data, dtype=torch.float32)
        data = self.complete_audio(data)
        return data, self.df.iloc[idx, 0]


def log(msg: str):
    """Helper function for timestamped logging - from your original"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# Global variable to store loaded config and model
_HYDRA_STATE = {'cfg': None, 'model': None, 'crop_frames': 208}


def _load_model_from_hydra(cfg: DictConfig, checkpoint_path: str, device: str):
    """Helper function to load model using Hydra config"""
    try:
        # Import register_resolvers if available
        from omegaconf import OmegaConf

        try:
            from src.utils import register_resolvers

            # Only register if not already done
            if not OmegaConf.has_resolver("effective_lr"):  # replace with one of yours
                register_resolvers()
                log("Custom resolvers registered")
            else:
                log("Resolvers already registered, skipping re-registration")

        except ImportError:
            log("Warning: Could not import register_resolvers, continuing without it")

        log("Instantiating model from config...")
        model = hydra.utils.instantiate(cfg.model)

        # Get crop_frames from config
        crop_frames = 208
        if hasattr(cfg.model, 'encoder') and hasattr(cfg.model.encoder, 'img_size'):
            crop_frames = cfg.model.encoder.img_size[1]
            log(f"Using crop_frames from config: {crop_frames}")

        # Load checkpoint weights
        log(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load state dict - handle both formats
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        model.to(device)

        log("Model loaded successfully!")
        
        return model, crop_frames

    except Exception as e:
        log(f"Error loading model: {e}")
        raise


# Register custom resolvers before Hydra initialization
#try:
 #   from src.utils.resolvers import register_resolvers
  #  register_resolvers()
  #  print("Custom resolvers registered before Hydra initialization")
#except ImportError:
 #   print("Warning: Could not import register_resolvers, continuing without it")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def _hydra_load_helper(cfg: DictConfig):
    """Top-level Hydra function to load config and store in global state"""
    _HYDRA_STATE['cfg'] = cfg


class EmbeddingExtractor:
    """
    Extract embeddings using pre-trained JEPA encoder
    Uses Hydra config - EXACTLY like your original precompute.py
    """

    def __init__(self, checkpoint_path, config_path, device=None):
        """
        Args:
            checkpoint_path: Path to model checkpoint (.ckpt file)
            config_path: Path to Hydra config directory (should contain train.yaml)
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        log(f"EmbeddingExtractor initialized:")
        log(f"  Checkpoint: {self.checkpoint_path}")
        log(f"  Config path: {self.config_path}")
        log(f"  Device: {self.device}")

        self.model = None
        self.cfg = None
        self.crop_frames = 208  # Default

    def load_model_with_hydra(self):
        """Load model using Hydra config - EXACTLY like your original precompute.py"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_path}")

        log("Loading model with Hydra configuration...")

        try:
            # Call the top-level Hydra function to load config
            log("Loading configuration with Hydra...")
            _hydra_load_helper()
            
            # Get config from global state
            self.cfg = _HYDRA_STATE['cfg']
            
            # Load model using the config
            self.model, self.crop_frames = _load_model_from_hydra(
                self.cfg, 
                str(self.checkpoint_path), 
                self.device
            )

        except Exception as e:
            log(f"Error loading model: {e}")
            raise

    def extract_embeddings(self, csv_file, data_dir, output_dir, batch_size=16, num_workers=4):
        """
        Extract embeddings for all files listed in CSV
        EXACTLY like your original precompute.py logic
        """
        if self.model is None:
            self.load_model_with_hydra()

        log(f"Extracting embeddings...")
        log(f"  Input CSV: {csv_file}")
        log(f"  Data directory: {data_dir}")
        log(f"  Output directory: {output_dir}")
        log(f"  Batch size: {batch_size}")

        # Create dataset and dataloader - EXACTLY like original
        dataset = MaskingEvalDataset(
            csv_file=csv_file,
            data_dir=data_dir,
            crop_frames=self.crop_frames,
            repeat_short=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        log(f"Dataset loaded with {len(dataset)} samples.")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Embeddings directory ready at {output_dir}")

        # Extract embeddings - EXACTLY like original
        with torch.no_grad():
            for i, (batch_data, filenames) in enumerate(dataloader):
                if i == 0:
                    log("Processing first batch...")

                batch_data = batch_data.to(self.device)

                # Get embeddings from encoder - EXACTLY like original
                batch_embeddings = self.model.encoder(batch_data)
                if isinstance(batch_embeddings, tuple):
                    batch_embeddings = batch_embeddings[0]
                batch_embeddings = batch_embeddings.cpu().numpy()

                # Save individual embeddings - EXACTLY like original
                for emb, fname in zip(batch_embeddings, filenames):
                    # Corrected logic to ensure directories exist
                    save_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_emb.npy")
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)

                    np.save(save_path, emb)

                # Progress logging
                if (i + 1) % 10 == 0:
                    log(f"Processed {i + 1} batches...")

        log(f"All embeddings saved to {output_dir}")

        return str(output_dir)


def extract_embeddings_for_masking_experiment(
    csv_file,
    data_dir,
    output_dir,
    checkpoint_path,
    config_path,
    batch_size=16,
    num_workers=4,
    device=None
):
    """
    Convenience function for embedding extraction in masking experiment

    Args:
        csv_file: Path to CSV file listing all input files
        data_dir: Base directory containing the .npy spectrogram files
        output_dir: Where to save embeddings
        checkpoint_path: Path to model checkpoint (e.g., last.ckpt)
        config_path: Path to Hydra config directory (containing train.yaml)
        batch_size: Batch size for processing
        num_workers: Number of data loader workers
        device: Device to use (cuda/cpu)
    """
    if not config_path:
        raise ValueError("config_path is required! Must point to directory containing train.yaml")

    extractor = EmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device
    )

    return extractor.extract_embeddings(
        csv_file=csv_file,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )


if __name__ == "__main__":
    # Test with your actual paths
    test_config = {
        'checkpoint_path': '/app/data/jepa_logs_subset10/xps/97d170e1/checkpoints/last.ckpt',
        'config_path': '/app/configs',  # Directory containing train.yaml
        'data_dir': '/app/data/test_spectrograms',
        'csv_file': '/app/data/test_files.csv',
        'output_dir': '/app/data/test_embeddings',
        'batch_size': 16
    }

    print("Testing embedding extraction...")
    try:
        result = extract_embeddings_for_masking_experiment(**test_config)
        print(f"Test successful! Embeddings saved to: {result}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
