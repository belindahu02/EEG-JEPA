"""
EEG CSV to log-mel spectrogram (LMS) converter with configurable overlap.
This program converts EEG .csv files found in the source folder to log-mel spectrograms,
then creates frames with configurable overlap from each channel's spectrogram.

AGGRESSIVE WINDOWING MODE (default): 20 second frames with 18 second overlap (90% overlap)

The conversion includes the following processes:
    - Multi-channel EEG processing (saves each RAW channel separately)
    - Resampling to target sampling rate
    - Converting to a log-mel spectrogram with same dimensions as audio converter
    - Segmenting into frames with configurable overlap

Input structure (flat directory with combined CSV files):
    combined/
        user1_same_combined.csv
        user1_diff_combined.csv
        user2_same_combined.csv
        user2_diff_combined.csv
        ...

Output structure (organized by user):
    output/
        user1/
            user1_same_combined/
                RAW_TP9_frame_000.npy
                RAW_TP9_frame_001.npy
                RAW_AF7_frame_000.npy
                ...
            user1_diff_combined/
                RAW_TP9_frame_000.npy
                ...
        user2/
            user2_same_combined/
                ...
            user2_diff_combined/
                ...
        files_audioset.csv

Example:
    # Default: 90% overlap
    python eeg_csv_to_lms_aggressive.py /path/to/combined /path/to/output
    
    # Custom overlap percentage
    python eeg_csv_to_lms_aggressive.py /path/to/combined /path/to/output --overlap_percent=95
    
    # Conservative overlap
    python eeg_csv_to_lms_aggressive.py /path/to/combined /path/to/output --overlap_percent=50
"""
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import fire
from tqdm import tqdm
import nnAudio.features
import warnings
from scipy import signal
import csv
import pandas as pd

warnings.simplefilter('ignore')


class EEG_FFT_parameters:
    # Match the audio converter parameters for consistent output dimensions
    sample_rate = 16000  # Target sampling rate (will resample EEG to this)
    window_size = 400  # 25ms window at 16kHz
    n_fft = 400
    hop_size = 160  # 10ms stride at 16kHz
    n_mels = 80  # Number of mel frequency bins
    f_min = 0.5  # Lower frequency limit for EEG (Hz)
    f_max = 100  # Upper frequency limit for EEG (Hz) - typical EEG range

    # Frame parameters - now configurable
    frame_duration = 20.0  # 20 seconds per frame
    overlap_percent = 90.0  # AGGRESSIVE: 90% overlap (was 50%)
    
    @property
    def frame_overlap(self):
        """Calculate overlap in seconds based on percentage"""
        return self.frame_duration * (self.overlap_percent / 100.0)
    
    @property
    def frame_stride(self):
        """Calculate stride in seconds"""
        return self.frame_duration - self.frame_overlap

    # CSV-specific parameters
    # Based on timestamp analysis, the original sampling rate appears to be ~2 Hz
    # This will be calculated per file based on timestamps
    original_sampling_rate = None  # Will be determined from data


def parse_timestamp(timestamp_str):
    """
    Parse timestamp in MM:SS.S format to seconds.
    
    Args:
        timestamp_str: String in format "MM:SS.S" or "MM:SS"
    
    Returns:
        Float value in seconds
    """
    parts = timestamp_str.split(':')
    minutes = float(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds


def estimate_sampling_rate(timestamps):
    """
    Estimate sampling rate from timestamp column.
    
    Args:
        timestamps: Array of timestamp strings
    
    Returns:
        Estimated sampling rate in Hz
    """
    if len(timestamps) < 2:
        return None
    
    # Convert first 10 timestamps to seconds
    times = [parse_timestamp(str(ts)) for ts in timestamps[:min(10, len(timestamps))]]
    
    # Calculate differences
    diffs = np.diff(times)
    
    # Average time difference
    avg_diff = np.mean(diffs)
    
    # Sampling rate is 1 / time_difference
    sampling_rate = 1.0 / avg_diff if avg_diff > 0 else None
    
    return sampling_rate


def create_overlapping_frames(spectrogram, frame_duration, frame_stride, hop_size, sample_rate):
    """
    Create overlapping frames from a spectrogram.

    Args:
        spectrogram: Input spectrogram tensor of shape (1, n_mels, time_frames)
        frame_duration: Duration of each frame in seconds
        frame_stride: Stride between frames in seconds
        hop_size: Hop size used in spectrogram computation
        sample_rate: Original sampling rate

    Returns:
        List of spectrogram frames, each of shape (1, n_mels, frame_time_frames)
    """
    batch_size, n_mels, total_time_frames = spectrogram.shape

    # Calculate frame dimensions in spectrogram time frames
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


def _converter_worker(args):
    subpathname, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose = args
    from_dir_path_obj, to_dir_path_obj = Path(from_dir), Path(to_dir)

    # Reconstruct the full path to the CSV file
    csv_input_path = from_dir_path_obj / subpathname

    # Create folder name from file name (without extension)
    file_stem = Path(subpathname).stem
    
    # Extract user identifier from filename (e.g., "user1" from "user1_same_combined.csv")
    # Assumes filenames start with "userN" or "userNN"
    import re
    match = re.match(r'^(user\d+)', file_stem)
    if match:
        user_id = match.group(1)
    else:
        # Fallback: no user prefix, just use filename
        user_id = "unknown_user"
    
    # Create output structure: to_dir/user_id/file_stem/
    folder_name = to_dir_path_obj / user_id / file_stem

    # Check if folder already exists and has files
    if folder_name.exists() and any(folder_name.glob('*.npy')):
        if verbose:
            print(f'Folder {file_stem} already exists with files at {folder_name}')
        return f'{file_stem} (already exists)'

    # Load and convert EEG to log-mel spectrogram
    try:
        # Load CSV data
        df = pd.read_csv(str(csv_input_path))
        
        # The column names in the CSV are Var1, Var2, etc.
        # Based on the column_headings.xlsx:
        # Var1 = TimeStamp
        # Var22 = RAW_TP9
        # Var23 = RAW_AF7
        # Var24 = RAW_AF8
        # Var25 = RAW_TP10
        
        # Extract timestamps and RAW channels
        timestamps = df['Var1'].values
        
        # RAW channels are at columns 22-25 (Var22-Var25)
        raw_channels = ['Var22', 'Var23', 'Var24', 'Var25']
        channel_names = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        
        # Estimate sampling rate from timestamps
        orig_sfreq = estimate_sampling_rate(timestamps)
        
        if orig_sfreq is None:
            print(f'ERROR: Could not estimate sampling rate for {csv_input_path.name}')
            return f'{file_stem} (sampling rate error)'
        
        if verbose:
            print(f'Processing {csv_input_path.name}: Estimated sampling rate: {orig_sfreq:.2f} Hz')

        # Create output folder
        folder_name.mkdir(parents=True, exist_ok=True)

        # Process each RAW channel separately
        successful_channels = []
        total_frames_saved = 0

        for channel_idx, (var_name, channel_name) in enumerate(zip(raw_channels, channel_names)):
            try:
                # Get channel data
                if var_name not in df.columns:
                    if verbose:
                        print(f'  Channel {var_name} not found in {csv_input_path.name}')
                    continue
                
                eeg_signal = df[var_name].values.astype(np.float32)
                
                # Remove any NaN or inf values
                eeg_signal = eeg_signal[np.isfinite(eeg_signal)]
                
                if len(eeg_signal) == 0:
                    if verbose:
                        print(f'  Channel {channel_name} has no valid data')
                    continue

                # Resample to target sampling rate if needed
                if abs(orig_sfreq - prms.sample_rate) > 0.1:
                    # Calculate resampling ratio
                    resample_ratio = prms.sample_rate / orig_sfreq
                    n_samples_new = int(len(eeg_signal) * resample_ratio)
                    eeg_signal = signal.resample(eeg_signal, n_samples_new)

                # Normalize EEG signal (important for spectrogram quality)
                eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)

                # Calculate minimum length for meaningful frames
                min_samples_for_frames = int(prms.sample_rate * prms.frame_duration)

                # Pad if too short for at least one frame
                if len(eeg_signal) < min_samples_for_frames:
                    if verbose:
                        print(f'Padding {subpathname} channel {channel_name} from {len(eeg_signal)} to {min_samples_for_frames} samples')
                    eeg_signal = np.pad(eeg_signal, (0, min_samples_for_frames - len(eeg_signal)))

                # Convert to log-mel spectrogram
                lms = to_lms(eeg_signal)

                # Create overlapping frames
                frames = create_overlapping_frames(
                    lms,
                    prms.frame_duration,
                    prms.frame_stride,
                    prms.hop_size,
                    prms.sample_rate
                )

                # Save each frame as a separate file
                channel_frames_saved = 0
                for frame_idx, frame in enumerate(frames):
                    frame_filename = folder_name / f"{channel_name}_frame_{frame_idx:03d}.npy"
                    np.save(frame_filename, frame.numpy())
                    channel_frames_saved += 1
                    total_frames_saved += 1

                if len(frames) > 0:
                    successful_channels.append(channel_name)
                    if verbose:
                        print(f'  Channel {channel_name} -> {channel_frames_saved} frames, each shape: {frames[0].shape}')
                else:
                    if verbose:
                        print(f'  Channel {channel_name} -> No frames generated (signal too short)')

            except Exception as e:
                print(f'ERROR processing channel {channel_name} in {subpathname}: {str(e)}')
                continue

        if verbose:
            print(f'Saved {total_frames_saved} total frames from {len(successful_channels)}/{len(channel_names)} channels for {csv_input_path.name}')

    except Exception as e:
        print('ERROR failed to open or convert', csv_input_path.name, '-', str(e))
        return f'{csv_input_path.name} (failed)'

    return f'{file_stem} ({len(successful_channels)}/{len(channel_names)} channels, {total_frames_saved} frames)'


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

    def __call__(self, signal):
        if isinstance(signal, np.ndarray):
            signal = torch.tensor(signal, dtype=torch.float32)

        if signal.dim() > 1:
            signal = signal.squeeze()

        x = self.to_spec(signal)
        x = (x + torch.finfo(torch.float32).eps).log()

        return x


def convert_eeg_batch(base_from_dir, base_to_dir,
                      overlap_percent=90.0, frame_duration=20.0,
                      suffix='.csv', skip=0, min_length=20.0, verbose=False) -> None:
    """
    Convert EEG CSV files to log-mel spectrograms with configurable frame overlap.

    Args:
        base_from_dir: Source directory containing CSV files (no subdirectories expected)
        base_to_dir: Base destination directory for .npy files
        overlap_percent: Percentage of overlap between frames (default: 90.0 for aggressive windowing)
                        - 50%: Conservative (10s stride with 20s frames) - original setting
                        - 75%: Moderate (5s stride) - 4x more samples
                        - 90%: Aggressive (2s stride) - 10x more samples (default)
                        - 95%: Very aggressive (1s stride) - 20x more samples
        frame_duration: Duration of each frame in seconds (default: 20.0)
        suffix: File extension to process (default: '.csv')
        skip: Number of files to skip (default: 0)
        min_length: Minimum length in seconds for processing (default: 20.0)
        verbose: Print detailed progress information
    """
    base_from_path = Path(base_from_dir)
    base_to_path = Path(base_to_dir)
    all_generated_file_paths = []

    prms = EEG_FFT_parameters()
    # Configure overlap
    prms.overlap_percent = overlap_percent
    prms.frame_duration = frame_duration
    
    to_lms = ToLogMelSpec(prms)

    # Calculate expected data multiplication
    original_stride = frame_duration * 0.5  # 50% overlap (original)
    new_stride = prms.frame_stride
    multiplication_factor = original_stride / new_stride

    print(f'Starting batch conversion for combined EEG dataset')
    print(f'=' * 70)
    print(f'Input Directory: {base_from_dir}')
    print(f'Output Directory: {base_to_dir}')
    print(f'Target sampling rate: {prms.sample_rate} Hz')
    print(f'\nAGGRESSIVE WINDOWING SETTINGS:')
    print(f'  Frame duration: {prms.frame_duration}s')
    print(f'  Overlap: {prms.overlap_percent}% ({prms.frame_overlap}s)')
    print(f'  Stride: {prms.frame_stride}s')
    print(f'  Expected data multiplication: ~{multiplication_factor:.1f}x more samples')
    print(f'    (compared to 50% overlap: {frame_duration * 0.5}s stride)')
    print(f'\nProcessing RAW channels (RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10)')
    print(f'Output shape per frame: (80, ~{int(prms.frame_duration * prms.sample_rate / prms.hop_size)})')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')
    print(f'=' * 70)

    if not base_from_path.exists():
        print(f"ERROR: Input directory not found: {base_from_path}")
        return

    # Find all CSV files directly in the input directory (no subdirectories)
    csv_files = list(base_from_path.glob(f'*{suffix}'))
    csv_files = sorted(csv_files)

    if skip > 0:
        csv_files = csv_files[skip:]

    if len(csv_files) == 0:
        print(f'No {suffix} files found in {base_from_path}')
        return

    print(f'\nFound {len(csv_files)} {suffix} files to process')

    # Prepare arguments for multiprocessing worker
    worker_args = []
    for csv_file in csv_files:
        # Use relative path (just the filename since there are no subdirectories)
        relative_path = csv_file.name
        worker_args.append([
            relative_path,
            str(base_from_path),
            str(base_to_path),
            prms,
            to_lms,
            suffix,
            min_length,
            verbose
        ])

    print(f'Processing {len(csv_files)} {suffix} files...')

    # Process files
    with Pool() as p:
        results = list(tqdm(p.imap(_converter_worker, worker_args), total=len(worker_args)))

    # Collect paths for CSV
    for subpathname, from_dir_worker, to_dir_worker, _, _, _, _, _ in worker_args:
        file_stem = Path(subpathname).stem
        
        # Extract user identifier to match the new structure
        import re
        match = re.match(r'^(user\d+)', file_stem)
        if match:
            user_id = match.group(1)
        else:
            user_id = "unknown_user"
        
        # Output folder is now: to_dir/user_id/file_stem/
        full_output_folder_path = Path(to_dir_worker) / user_id / file_stem

        if full_output_folder_path.exists():
            for npy_file in full_output_folder_path.glob('*.npy'):
                relative_path_from_base_output = npy_file.relative_to(base_to_path)
                all_generated_file_paths.append(relative_path_from_base_output)

    successful_files = [r for r in results if r and 'failed' not in r]
    print(f'\n{"=" * 70}')
    print(f'Finished processing. Successfully processed {len(successful_files)}/{len(csv_files)} files.')
    
    if verbose:
        print(f'\nDetailed results:')
        for result in results:
            if result:
                print(f'  {result}')

    # Write to CSV
    all_generated_file_paths.sort()

    csv_filename = base_to_path / "files_audioset.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['file_name'])
            for filepath in all_generated_file_paths:
                csv_writer.writerow([str(filepath)])
        print(f"\n{'=' * 70}")
        print(f"Successfully wrote ALL generated file paths to {csv_filename}")
        print(f"Total files generated: {len(all_generated_file_paths)}")
        print(f"With {prms.overlap_percent}% overlap, you have ~{multiplication_factor:.1f}x more training samples!")
        print(f"{'=' * 70}")
    except Exception as e:
        print(f"ERROR writing to CSV file {csv_filename}: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    fire.Fire(convert_eeg_batch)
