"""
MusicID utilities with CSV-to-spectrogram conversion
CORRECTED: Handles flat directory structure with augmented files
Filename format: userX_TYPE_sessionY[_aug_ZZZ].csv
"""

import numpy as np
import torch
from pathlib import Path
import warnings
import os
import csv
import pandas as pd
from scipy import signal
import nnAudio.features
from tqdm import tqdm
import re

warnings.simplefilter('ignore')


class EEG_FFT_parameters:
    """Match the audio converter parameters for consistent output dimensions"""
    sample_rate = 16000  # Target sampling rate (will resample EEG to this)
    window_size = 400  # 25ms window at 16kHz
    n_fft = 400
    hop_size = 160  # 10ms stride at 16kHz
    n_mels = 80  # Number of mel frequency bins
    f_min = 0.5  # Lower frequency limit for EEG (Hz)
    f_max = 100  # Upper frequency limit for EEG (Hz)

    # Frame parameters
    frame_duration = 20.0  # 20 seconds per frame
    frame_overlap = 10.0   # 10 seconds overlap
    frame_stride = frame_duration - frame_overlap  # 10 seconds stride


class ToLogMelSpec:
    def __init__(self, cfg):
        self.cfg = cfg
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def __call__(self, signal_input):
        if isinstance(signal_input, np.ndarray):
            signal_input = torch.tensor(signal_input, dtype=torch.float32)

        if signal_input.dim() > 1:
            signal_input = signal_input.squeeze()

        x = self.to_spec(signal_input)
        x = (x + torch.finfo(torch.float32).eps).log()

        return x


def parse_musicid_filename(filename):
    """
    Parse MusicID augmented filename
    
    Format: userX_TYPE_sessionY[_aug_ZZZ].csv
    Examples:
        - user10_fav_session1.csv -> user=10, type=fav, session=1, aug=None
        - user19_same_session1_aug_001.csv -> user=19, type=same, session=1, aug=1
    
    Returns:
        dict with keys: user_id, session_type, session_num, aug_num (or None)
    """
    stem = Path(filename).stem  # Remove .csv
    
    # Pattern: userX_TYPE_sessionY[_aug_ZZZ]
    pattern = r'user(\d+)_(\w+)_session(\d+)(?:_aug_(\d+))?'
    match = re.match(pattern, stem)
    
    if not match:
        return None
    
    user_id = int(match.group(1))
    session_type = match.group(2)  # 'fav', 'same', etc.
    session_num = int(match.group(3))
    aug_num = int(match.group(4)) if match.group(4) else None
    
    return {
        'user_id': user_id,
        'session_type': session_type,
        'session_num': session_num,
        'aug_num': aug_num,
        'is_augmented': aug_num is not None
    }


def parse_timestamp(timestamp_str):
    """
    Parse timestamp to seconds. Handles multiple formats:
    - HH:MM:SS.S (hours:minutes:seconds)
    - MM:SS.S (minutes:seconds)
    - SS.S (just seconds)
    """
    parts = str(timestamp_str).split(':')
    
    if len(parts) == 3:
        # HH:MM:SS.S format
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # MM:SS.S format
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 1:
        # Just seconds
        return float(parts[0])
    else:
        raise ValueError(f"Unrecognized timestamp format: {timestamp_str}")


def estimate_sampling_rate(timestamps):
    """Estimate sampling rate from timestamp column"""
    if len(timestamps) < 2:
        return None
    
    times = [parse_timestamp(str(ts)) for ts in timestamps[:min(10, len(timestamps))]]
    diffs = np.diff(times)
    avg_diff = np.mean(diffs)
    sampling_rate = 1.0 / avg_diff if avg_diff > 0 else None
    
    return sampling_rate


def create_overlapping_frames(spectrogram, frame_duration, frame_stride, hop_size, sample_rate):
    """Create overlapping frames from a spectrogram"""
    batch_size, n_mels, total_time_frames = spectrogram.shape

    frames_per_second = sample_rate / hop_size
    frame_length_frames = int(frame_duration * frames_per_second)
    frame_stride_frames = int(frame_stride * frames_per_second)

    frames = []
    start_frame = 0

    while start_frame + frame_length_frames <= total_time_frames:
        end_frame = start_frame + frame_length_frames
        frame = spectrogram[:, :, start_frame:end_frame]
        frames.append(frame)
        start_frame += frame_stride_frames

    return frames


class MusicIDMaskingProcessor:
    """Process MusicID CSV files: convert to spectrograms, then apply masking"""

    def __init__(self):
        self.prms = EEG_FFT_parameters()
        self.to_lms = ToLogMelSpec(self.prms)
        print(f"MusicID Masking Processor initialized")
        print(f"  Converting CSV to spectrograms with {self.prms.sample_rate} Hz")
        print(f"  Frame duration: {self.prms.frame_duration}s with {self.prms.frame_overlap}s overlap")

    def convert_csv_to_spectrogram(self, csv_path):
        """
        Convert a single CSV file to spectrogram frames
        
        Returns:
            List of spectrogram frames (each frame is a tensor)
        """
        df = pd.read_csv(str(csv_path))
        
        # Column mapping for MusicID:
        # Var1 = TimeStamp
        # Var22-25 = RAW channels (TP9, AF7, AF8, TP10)
        timestamps = df['Var1'].values
        
        # Estimate sampling rate
        orig_sfreq = estimate_sampling_rate(timestamps)
        if orig_sfreq is None:
            raise ValueError(f"Could not estimate sampling rate for {csv_path.name}")
        
        # Process all RAW channels and stack them
        raw_channels = ['Var22', 'Var23', 'Var24', 'Var25']
        channel_spectrograms = []
        
        for var_name in raw_channels:
            if var_name not in df.columns:
                continue
            
            eeg_signal = df[var_name].values.astype(np.float32)
            eeg_signal = eeg_signal[np.isfinite(eeg_signal)]
            
            if len(eeg_signal) == 0:
                continue
            
            # Resample if needed
            if abs(orig_sfreq - self.prms.sample_rate) > 0.1:
                resample_ratio = self.prms.sample_rate / orig_sfreq
                n_samples_new = int(len(eeg_signal) * resample_ratio)
                eeg_signal = signal.resample(eeg_signal, n_samples_new)
            
            # Normalize
            eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)
            
            # Pad if too short
            min_samples = int(self.prms.sample_rate * self.prms.frame_duration)
            if len(eeg_signal) < min_samples:
                eeg_signal = np.pad(eeg_signal, (0, min_samples - len(eeg_signal)))
            
            # Convert to log-mel spectrogram
            lms = self.to_lms(eeg_signal)
            # Squeeze out batch dimension if present (nnAudio outputs (1, n_mels, time))
            if lms.dim() == 3:
                lms = lms.squeeze(0)  # Now (n_mels, time)
            channel_spectrograms.append(lms)
        
        if not channel_spectrograms:
            raise ValueError(f"No valid channels found in {csv_path.name}")
        
        # Stack all channels (create multi-channel spectrogram)
        stacked_spec = torch.stack(channel_spectrograms, dim=0)  # Shape: (n_channels, n_mels, time)
        
        # Ensure we have exactly 3 dimensions (create_overlapping_frames expects 3D)
        if stacked_spec.dim() == 4:
            stacked_spec = stacked_spec.squeeze(1)  # Remove extra dimension
        
        # Note: create_overlapping_frames will treat first dim as "batch" (our n_channels)
        
        # Create overlapping frames
        frames = create_overlapping_frames(
            stacked_spec,
            self.prms.frame_duration,
            self.prms.frame_stride,
            self.prms.hop_size,
            self.prms.sample_rate
        )
        
        return frames

    def apply_masking(self, spectrogram, masking_percentage, num_blocks):
        """
        Apply time-domain masking to spectrogram
        
        Args:
            spectrogram: numpy array or tensor of shape (channels, freq, time)
            masking_percentage: percentage of time frames to mask (0-100)
            num_blocks: number of contiguous blocks to mask
        
        Returns:
            Masked spectrogram
        """
        if masking_percentage == 0:
            if isinstance(spectrogram, torch.Tensor):
                return spectrogram.clone()
            return spectrogram.copy()
        
        if isinstance(spectrogram, torch.Tensor):
            spec_masked = spectrogram.clone()
        else:
            spec_masked = spectrogram.copy()
        
        # Get time dimension
        if spec_masked.ndim == 3:
            _, _, time_frames = spec_masked.shape
        elif spec_masked.ndim == 2:
            _, time_frames = spec_masked.shape
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spec_masked.shape}")
        
        # Calculate total frames to mask
        total_frames_to_mask = int(time_frames * masking_percentage / 100)
        
        if total_frames_to_mask == 0:
            return spec_masked
        
        # Calculate frames per block
        frames_per_block = max(1, total_frames_to_mask // num_blocks)
        
        # Apply masking in blocks
        for _ in range(num_blocks):
            if frames_per_block >= time_frames:
                # Mask everything
                spec_masked[:] = 0
                break
            
            # Random start position for this block
            start_idx = np.random.randint(0, max(1, time_frames - frames_per_block))
            end_idx = min(start_idx + frames_per_block, time_frames)
            
            # Apply mask
            if spec_masked.ndim == 3:
                spec_masked[:, :, start_idx:end_idx] = 0
            else:
                spec_masked[:, start_idx:end_idx] = 0
        
        return spec_masked

    def process_csv_file(self, csv_path, masking_percentage=0, num_blocks=1, output_dir=None):
        """
        Convert CSV to spectrograms and apply masking
        
        Args:
            csv_path: Path to CSV file
            masking_percentage: Percentage of time frames to mask (0-100)
            num_blocks: Number of contiguous masking blocks
            output_dir: Where to save processed spectrograms
        
        Returns:
            Number of frames processed
        """
        csv_path = Path(csv_path)
        
        # Convert CSV to spectrogram frames
        frames = self.convert_csv_to_spectrogram(csv_path)
        
        # Apply masking to each frame
        masked_frames = []
        for frame in frames:
            masked_frame = self.apply_masking(frame, masking_percentage, num_blocks)
            masked_frames.append(masked_frame)
        
        # Save frames if output_dir provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, frame in enumerate(masked_frames):
                # Convert to numpy if tensor
                if isinstance(frame, torch.Tensor):
                    frame_np = frame.numpy()
                else:
                    frame_np = frame
                
                # Save with _stacked naming convention
                frame_filename = output_dir / f"{i:03d}_stacked.npy"
                np.save(frame_filename, frame_np)
        
        return len(masked_frames)


def create_masked_dataset_for_experiment(raw_csv_dir, output_dir, user_ids, session_nums,
                                        masking_percentage, num_blocks, 
                                        use_augmented=True, session_types=None):
    """
    Create a complete masked dataset for the MusicID experiment from CSV files
    
    Args:
        raw_csv_dir: Flat directory with MusicID CSV files (/media/.../musicid_augmented)
        output_dir: Where to save processed spectrograms
        user_ids: List of user IDs to process (e.g., [4, 10, 19])
        session_nums: List of session numbers to process (e.g., [1, 2])
        masking_percentage: Percentage to mask (0-50)
        num_blocks: Number of masking blocks (1-5)
        use_augmented: If True, include augmented files (_aug_XXX)
        session_types: List of session types to include (e.g., ['fav', 'same']), None = all
    """
    processor = MusicIDMaskingProcessor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_dir = Path(raw_csv_dir)
    
    print(f"Creating masked MusicID dataset from CSV files:")
    print(f"  Input CSV directory: {raw_csv_dir}")
    print(f"  Users: {user_ids} ({len(user_ids)} users)")
    print(f"  Sessions: {session_nums}")
    print(f"  Session types: {session_types or 'all'}")
    print(f"  Use augmented: {use_augmented}")
    print(f"  Masking: {masking_percentage}% with {num_blocks} blocks")
    print(f"  Output: {output_dir}")
    
    # Find all CSV files matching criteria
    all_csv_files = list(raw_csv_dir.glob("*.csv"))
    
    print(f"Found {len(all_csv_files)} total CSV files in directory")
    
    selected_files = []
    for csv_file in all_csv_files:
        info = parse_musicid_filename(csv_file.name)
        if info is None:
            continue
        
        # Filter by user
        if info['user_id'] not in user_ids:
            continue
        
        # Filter by session number
        if info['session_num'] not in session_nums:
            continue
        
        # Filter by session type
        if session_types is not None and info['session_type'] not in session_types:
            continue
        
        # Filter by augmented status
        if not use_augmented and info['is_augmented']:
            continue
        
        selected_files.append((csv_file, info))
    
    print(f"Selected {len(selected_files)} files matching criteria")
    
    if len(selected_files) == 0:
        print("WARNING: No files matched the selection criteria!")
        return str(output_dir)
    
    # Group files for processing
    files_by_user_session = {}
    for csv_file, info in selected_files:
        key = (info['user_id'], info['session_type'], info['session_num'])
        if key not in files_by_user_session:
            files_by_user_session[key] = []
        files_by_user_session[key].append((csv_file, info))
    
    print(f"Processing {len(files_by_user_session)} unique user-session combinations")
    
    successful_files = 0
    total_frames = 0
    all_output_files = []
    
    for (user_id, session_type, session_num), file_list in tqdm(files_by_user_session.items()):
        # Create output structure: output_dir/userX/userX_TYPE_sessionY/
        user_folder = f"user{user_id}"
        session_folder = f"user{user_id}_{session_type}_session{session_num}"
        session_output_dir = output_dir / user_folder / session_folder
        
        print(f"\nProcessing user{user_id} {session_type} session{session_num} ({len(file_list)} files)")
        
        for csv_file, info in file_list:
            try:
                # Create subdirectory for augmented files
                if info['is_augmented']:
                    file_output_dir = session_output_dir / f"aug_{info['aug_num']:03d}"
                else:
                    file_output_dir = session_output_dir / "original"
                
                # Convert and mask
                num_frames = processor.process_csv_file(
                    csv_path=csv_file,
                    masking_percentage=masking_percentage,
                    num_blocks=num_blocks,
                    output_dir=file_output_dir
                )
                
                successful_files += 1
                total_frames += num_frames
                
                # Collect output file paths for CSV
                for npy_file in file_output_dir.glob("*.npy"):
                    rel_path = npy_file.relative_to(output_dir)
                    all_output_files.append(str(rel_path))
                
                print(f"  ✓ {csv_file.name}: {num_frames} frames")
                
            except Exception as e:
                print(f"  ✗ {csv_file.name}: Error - {e}")
    
    # Create CSV file listing all generated files
    csv_path = output_dir / "files_audioset.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name'])
        for file_path in sorted(all_output_files):
            writer.writerow([file_path])
    
    print(f"\nDataset creation completed:")
    print(f"  Successful CSV files: {successful_files}/{len(selected_files)}")
    print(f"  Total frames generated: {total_frames}")
    print(f"  Output files: {len(all_output_files)}")
    print(f"  CSV listing: {csv_path}")
    
    return str(output_dir)


def run_embedding_extraction(csv_path, data_dir, embeddings_dir, model_checkpoint_path,
                           config_path, batch_size=16, num_workers=4, device=None):
    """Extract embeddings using a pre-trained JEPA encoder model"""
    from embeddings import extract_embeddings_for_masking_experiment
    
    print(f"Extracting embeddings from {data_dir} to {embeddings_dir}")
    print(f"Using checkpoint: {model_checkpoint_path}")
    print(f"Using config: {config_path}")
    
    if not model_checkpoint_path or not Path(model_checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint_path}")
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")
    
    return extract_embeddings_for_masking_experiment(
        csv_file=csv_path,
        data_dir=data_dir,
        output_dir=embeddings_dir,
        checkpoint_path=model_checkpoint_path,
        config_path=config_path,
        batch_size=batch_size,
        device=device
    )


def run_embedding_grouping(embeddings_dir, grouped_dir):
    """Group embeddings by frame ID - adapted for MusicID structure"""
    print(f"Grouping embeddings from {embeddings_dir} to {grouped_dir}")
    
    embeddings_dir = Path(embeddings_dir)
    grouped_dir = Path(grouped_dir)
    grouped_dir.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    
    for root, dirs, files in os.walk(embeddings_dir):
        # Only process leaf folders with .npy files
        npy_files = [f for f in files if f.endswith("_emb.npy")]
        if not npy_files:
            continue
        
        # Create corresponding output directory
        leaf_output_dir = grouped_dir / Path(root).relative_to(embeddings_dir)
        leaf_output_dir.mkdir(parents=True, exist_ok=True)
        
        for f in npy_files:
            # Load embedding
            emb = np.load(Path(root) / f)
            
            # Save with "_stacked.npy" name (remove "_emb" suffix)
            output_name = f.replace("_emb.npy", "_stacked.npy")
            
            save_path = leaf_output_dir / output_name
            np.save(save_path, emb)
            total_files += 1
        
        if total_files % 100 == 0:
            print(f"  Processed {total_files} files...")
    
    print(f"Embedding grouping completed: {total_files} total files")
    return str(grouped_dir)


if __name__ == "__main__":
    # Test filename parsing
    test_filenames = [
        "user10_fav_session1.csv",
        "user19_same_session1_aug_001.csv",
        "user4_fav_session5_aug_010.csv"
    ]
    
    print("Testing filename parsing:")
    for fn in test_filenames:
        info = parse_musicid_filename(fn)
        print(f"  {fn}")
        print(f"    -> {info}")
