"""
Converts EEG .edf files from the PhysioNet Motor Movement/Imagery Dataset
to log-mel spectrograms, then creates 20-second frames with 10-second overlap from each 
channel's spectrogram. Each frame is saved as a separate .npy file.

The conversion includes the following processes:
    - Multi-channel EEG processing (saves each channel separately)
    - Augmentation (frequency wrapping + amplitude scaling)
    - Resampling to target sampling rate
    - Converting to a log-mel spectrogram with same dimensions as audio converter
    - Segmenting into 20s frames with 10s overlap

Input structure (PhysioNet EEG Motor Movement/Imagery Dataset):
    base_dir/
        S001/
            S001R01.edf
            S001R02.edf
            ...
            S001R14.edf
        S002/
            S002R01.edf
            ...
        ...
        S109/
            S109R01.edf
            ...

Output structure:
    output_dir/
        S001/
            S001R01/
                Fc5._frame_000.npy
                Fc5._frame_001.npy
                Fc3._frame_000.npy
                ...
            S001R02/
                ...
        S002/
            ...
        files_audioset.csv

Example usage:
    # WITH augmentation (recommended for improving accuracy):
    python eeg_to_lms_augmented.py /path/to/physionet /path/to/output \
        --start_subject_id=1 --end_subject_id=109 --augment=True
    
    # WITHOUT augmentation (baseline):
    python eeg_to_lms_augmented.py /path/to/physionet /path/to/output \
        --start_subject_id=1 --end_subject_id=109 --augment=False
    
    # Process only subjects S001-S010:
    python eeg_to_lms_augmented.py /path/to/physionet /path/to/output \
        --start_subject_id=1 --end_subject_id=10 --augment=True
    
    # With reproducible random seed:
    python eeg_to_lms_augmented.py /path/to/physionet /path/to/output \
        --start_subject_id=1 --end_subject_id=109 --augment=True --seed=42
    
    # Resume interrupted run (automatically detected):
    python eeg_to_lms_augmented.py /path/to/physionet /path/to/output \
        --start_subject_id=1 --end_subject_id=109 --augment=True
"""
import numpy as np
from pathlib import Path
import mne
from multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import fire
from tqdm import tqdm
import nnAudio.features
import warnings
from scipy import signal
import csv

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

    # Frame parameters
    frame_duration = 20.0  # 20 seconds per frame
    frame_overlap = 10.0   # 10 seconds overlap
    frame_stride = frame_duration - frame_overlap  # 10 seconds stride
    
    # Augmentation parameters
    augment = True  # Enable on-the-fly augmentation
    time_scale_range = (0.8, 1.2)  # Frequency wrapping range
    amplitude_scale_range = (0.8, 1.2)  # Amplitude scaling range


def scan_existing_output_folders(base_output_dir):
    """
    Scan the output directory to find all folders that already have .npy files.
    
    Returns:
        set: Set of folder paths relative to base that have been processed (e.g., 'S001/S001R01')
    """
    processed_folders = set()
    base_path = Path(base_output_dir)
    
    if not base_path.exists():
        return processed_folders
    
    # Look for all folders containing .npy files
    # Pattern: base_output_dir/S001/S001R01/*.npy
    for subject_dir in base_path.glob('S*'):
        if subject_dir.is_dir():
            for run_dir in subject_dir.glob('*'):
                if run_dir.is_dir():
                    # Check if this folder has any .npy files
                    if any(run_dir.glob('*.npy')):
                        relative_path = run_dir.relative_to(base_path)
                        processed_folders.add(str(relative_path))
    
    return processed_folders


def get_most_recent_processed_folder(processed_folders, base_output_dir):
    """
    Get the most recently modified folder to redo it (in case of corruption).
    
    Returns:
        str or None: The most recent folder path to redo
    """
    if not processed_folders:
        return None
    
    base_path = Path(base_output_dir)
    
    # Find the folder with the most recent modification time
    most_recent = None
    most_recent_time = 0
    
    for folder_path in processed_folders:
        full_path = base_path / folder_path
        if full_path.exists():
            mtime = full_path.stat().st_mtime
            if mtime > most_recent_time:
                most_recent_time = mtime
                most_recent = folder_path
    
    return most_recent


def apply_augmentation(eeg_signal, time_scale_factor, amplitude_scale_factor):
    """
    Apply on-the-fly augmentation to EEG signal.
    
    Args:
        eeg_signal: 1D numpy array of EEG samples
        time_scale_factor: Time axis scaling factor (frequency wrapping)
        amplitude_scale_factor: Amplitude scaling factor
    
    Returns:
        Augmented EEG signal
    """
    # Apply frequency wrapping (time scaling)
    n_samples_original = len(eeg_signal)
    n_samples_new = int(n_samples_original / time_scale_factor)
    eeg_augmented = signal.resample(eeg_signal, n_samples_new)
    
    # Apply amplitude scaling
    eeg_augmented = eeg_augmented * amplitude_scale_factor
    
    return eeg_augmented


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

    # Reconstruct the full path to the EDF file
    edf_input_path = from_dir_path_obj / subpathname

    # Create folder name from file name (without extension)
    file_stem = Path(subpathname).stem
    relative_output_path = Path(subpathname).parent / file_stem
    folder_name = to_dir_path_obj / relative_output_path

    # Check if folder already exists and has files
    if folder_name.exists() and any(folder_name.glob('*.npy')):
        if verbose:
            print(f'Folder {file_stem} already exists with files at {folder_name}')
        return f'{file_stem} (already exists)'

    # Load and convert EEG to log-mel spectrogram
    try:
        # Load EEG data
        raw = mne.io.read_raw_edf(str(edf_input_path), preload=True, verbose=False)

        # Get EEG data and channel names
        eeg_data = raw.get_data()  # Shape: (n_channels, n_samples)
        channel_names = raw.ch_names
        orig_sfreq = raw.info['sfreq']
        n_channels = eeg_data.shape[0]

        if verbose:
            augment_status = "WITH augmentation" if prms.augment else "WITHOUT augmentation"
            print(f'Processing {edf_input_path.name}: {n_channels} channels ({augment_status})')

        # Create output folder
        folder_name.mkdir(parents=True, exist_ok=True)

        # Process each channel separately
        successful_channels = []
        total_frames_saved = 0

        for channel_idx in range(n_channels):
            try:
                # Get channel data
                eeg_signal = eeg_data[channel_idx, :]
                channel_name = channel_names[channel_idx].replace(' ', '').replace('.', '_')

                # Apply on-the-fly augmentation if enabled
                if prms.augment:
                    time_scale_factor = np.random.uniform(*prms.time_scale_range)
                    amplitude_scale_factor = np.random.uniform(*prms.amplitude_scale_range)
                    eeg_signal = apply_augmentation(eeg_signal, time_scale_factor, amplitude_scale_factor)

                # Resample to target sampling rate
                if orig_sfreq != prms.sample_rate:
                    n_samples_new = int(len(eeg_signal) * prms.sample_rate / orig_sfreq)
                    eeg_signal = signal.resample(eeg_signal, n_samples_new)

                # Check minimum length for at least one frame
                min_samples_for_frames = int(prms.frame_duration * prms.sample_rate)
                if len(eeg_signal) < min_samples_for_frames:
                    if verbose:
                        print(f'Padding {subpathname} channel {channel_idx} from {len(eeg_signal)} to {min_samples_for_frames} samples')
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
                        print(f'  Channel {channel_idx} ({channel_name}) -> {channel_frames_saved} frames, each shape: {frames[0].shape}')
                else:
                    if verbose:
                        print(f'  Channel {channel_idx} ({channel_name}) -> No frames generated (signal too short)')

            except Exception as e:
                print(f'ERROR processing channel {channel_idx} in {subpathname}: {str(e)}')
                continue

        if verbose:
            print(f'Saved {total_frames_saved} total frames from {len(successful_channels)}/{n_channels} channels for {edf_input_path.name}')

    except Exception as e:
        print('ERROR failed to open or convert', edf_input_path.name, '-', str(e))
        return f'{edf_input_path.name} (failed)'

    return f'{file_stem} ({len(successful_channels)}/{n_channels} channels, {total_frames_saved} frames)'


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
                      start_subject_id=1, end_subject_id=109,
                      suffix='.edf', skip=0, min_length=20.0, 
                      augment=True, seed=None, verbose=False) -> None:
    """
    Convert EEG files from a range of subjects to log-mel spectrograms with 20s frames, 10s overlap, and on-the-fly augmentation.
    Supports resuming from interrupted runs.

    Args:
        base_from_dir: Base source directory (e.g., /workspace/input_eeg_data/physionet.org/files/eegmmidb/1.0.0/)
        base_to_dir: Base destination directory for .npy files (e.g., /workspace/output_spectrograms)
        start_subject_id: Starting subject ID (inclusive, e.g., 1 for S001)
        end_subject_id: Ending subject ID (inclusive, e.g., 109 for S109)
        suffix: File extension to process (default: '.edf')
        skip: Number of files to skip per subject (default: 0)
        min_length: Minimum length in seconds for processing (default: 20.0 for at least one frame)
        augment: Enable on-the-fly augmentation (default: True)
        seed: Random seed for augmentation reproducibility (default: None)
        verbose: Print detailed progress information
    """
    if seed is not None:
        np.random.seed(seed)
    
    base_from_path = Path(base_from_dir)
    base_to_path = Path(base_to_dir)
    csv_filename = base_to_path / "files_audioset.csv"
    
    # Check for existing output folders
    print(f"\nScanning for existing output folders...")
    print(f"Output directory: {base_to_dir}")
    
    processed_folders = scan_existing_output_folders(base_to_dir)
    most_recent_folder = get_most_recent_processed_folder(processed_folders, base_to_dir)
    
    if processed_folders:
        print(f"\n{'='*60}")
        print(f"RESUME MODE DETECTED")
        print(f"{'='*60}")
        print(f"Found {len(processed_folders)} already processed folder(s)")
        if most_recent_folder:
            print(f"Most recent folder: {most_recent_folder}")
            print(f"Will REDO the most recent folder to ensure completeness")
            # Remove the most recent folder so it gets reprocessed
            processed_folders.discard(most_recent_folder)
            # Also delete the folder contents to ensure clean reprocessing
            folder_to_delete = base_to_path / most_recent_folder
            if folder_to_delete.exists():
                import shutil
                print(f"Deleting {most_recent_folder} for reprocessing...")
                shutil.rmtree(folder_to_delete)
        print(f"Skipping {len(processed_folders)} completed folder(s)")
        print(f"{'='*60}\n")
    else:
        print(f"✓ No previous progress detected - starting from beginning\n")
    
    all_generated_file_paths = []

    prms = EEG_FFT_parameters()
    prms.augment = augment  # Set augmentation flag
    to_lms = ToLogMelSpec(prms)

    print(f'Starting batch conversion for subjects S{start_subject_id:03d} to S{end_subject_id:03d}')
    print(f'Base Input Directory: {base_from_dir}')
    print(f'Base Output Directory: {base_to_dir}')
    print(f'Target sampling rate: {prms.sample_rate} Hz')
    print(f'Processing ALL channels separately with 20s frames and 10s overlap')
    print(f'Frame duration: {prms.frame_duration}s, Frame stride: {prms.frame_stride}s')
    print(f'Output shape per frame: (80, ~{int(prms.frame_duration * prms.sample_rate / prms.hop_size)})')
    print(f'Frequency range: {prms.f_min}-{prms.f_max} Hz')
    
    # Augmentation info
    if augment:
        print(f'ON-THE-FLY AUGMENTATION: ENABLED')
        print(f'  - Frequency wrapping (time scaling): U{prms.time_scale_range}')
        print(f'  - Amplitude scaling: U{prms.amplitude_scale_range}')
        if seed is not None:
            print(f'  - Random seed: {seed}')
    else:
        print(f'ON-THE-FLY AUGMENTATION: DISABLED')

    total_skipped = 0
    total_processed = 0

    for i in range(start_subject_id, end_subject_id + 1):
        subject_id = f"S{i:03d}"  # Format as S001, S002, etc.

        # Construct the full source and destination paths for the current subject
        # Input: base_from_dir/S001/...edf
        # Output: base_to_dir/S001/...npy
        current_subject_from_dir = base_from_path / subject_id
        current_subject_to_dir = base_to_path / subject_id

        if not current_subject_from_dir.exists():
            print(f"WARNING: Source directory for {subject_id} not found: {current_subject_from_dir}. Skipping.")
            continue

        print(f"\n--- Processing Subject: {subject_id} ---")

        # Find all EDF files for the current subject
        subject_edf_files = [f.relative_to(current_subject_from_dir) for f in current_subject_from_dir.glob(f'**/*{suffix}')]
        subject_edf_files = sorted(subject_edf_files)

        if skip > 0:
            subject_edf_files = subject_edf_files[skip:]

        if len(subject_edf_files) == 0:
            print(f'No {suffix} files found for {subject_id} in {current_subject_from_dir}')
            continue

        # Filter out already processed files
        files_to_process = []
        subject_skipped = 0
        
        for f in subject_edf_files:
            file_stem = Path(f).stem
            # The relative path within the subject folder
            relative_output_path = Path(f).parent / file_stem
            # Full identifier including subject (e.g., "S001/S001R01")
            # f is relative to subject dir, so we need to prepend subject_id
            full_folder_identifier = str(Path(subject_id) / relative_output_path)
            
            if full_folder_identifier in processed_folders:
                subject_skipped += 1
                total_skipped += 1
            else:
                files_to_process.append(f)
        
        if subject_skipped > 0:
            print(f'  Skipping {subject_skipped} already processed file(s)')
        
        if len(files_to_process) == 0:
            print(f'  All files already processed for {subject_id}')
            continue

        # Prepare arguments for multiprocessing worker for this subject's files
        worker_args_for_subject = []
        for f in files_to_process:
            worker_args_for_subject.append([f, str(current_subject_from_dir), str(current_subject_to_dir), prms, to_lms, suffix, min_length, verbose])

        print(f'  Processing {len(files_to_process)} {suffix} file(s) for {subject_id}...')

        # Process files for this subject
        with Pool() as p:
            results = list(tqdm(p.imap(_converter_worker, worker_args_for_subject), total=len(worker_args_for_subject)))

        total_processed += len(files_to_process)

        # Collect paths for CSV for the current subject
        for subpathname_relative_to_subject_dir, from_dir_worker, to_dir_worker, _, _, _, _, _ in worker_args_for_subject:
            file_stem = Path(subpathname_relative_to_subject_dir).stem
            output_subfolder_name = Path(subpathname_relative_to_subject_dir).parent / file_stem
            full_output_folder_path_in_container = Path(to_dir_worker) / output_subfolder_name

            if full_output_folder_path_in_container.exists():
                for npy_file in full_output_folder_path_in_container.glob('*.npy'):
                    relative_path_from_base_output = npy_file.relative_to(base_to_path)
                    all_generated_file_paths.append(relative_path_from_base_output)

        successful_subject_files = [r for r in results if r and 'failed' not in r]
        print(f'  Finished {subject_id}. Successfully processed {len(successful_subject_files)}/{len(files_to_process)} file(s).')
        if verbose:
            for result in results:
                if result:
                    print(f'    {result}')

    # Append to CSV (or create if it doesn't exist)
    all_generated_file_paths.sort()

    try:
        # Read existing entries
        existing_entries = set()
        if csv_filename.exists():
            with open(csv_filename, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)  # Skip header
                for row in csv_reader:
                    if row:
                        existing_entries.add(row[0])
        
        # Combine existing and new entries
        all_entries = sorted(existing_entries.union(set(str(fp) for fp in all_generated_file_paths)))
        
        # Write combined list
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['file_name'])
            for filepath in all_entries:
                csv_writer.writerow([filepath])
        
        print(f"\n{'='*60}")
        print(f"Successfully updated {csv_filename}")
        print(f"Total files in CSV: {len(all_entries)}")
        print(f"Files added in this run: {len(all_generated_file_paths)}")
        print(f"Files skipped (already processed): {total_skipped}")
        print(f"Files processed in this run: {total_processed}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"ERROR writing to CSV file {csv_filename}: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    fire.Fire(convert_eeg_batch)
