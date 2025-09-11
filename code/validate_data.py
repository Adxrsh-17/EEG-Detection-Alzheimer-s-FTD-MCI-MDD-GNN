import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm
import sys

# --- Configuration ---
METADATA_PATH = Path("../data/harmonized_metadata.csv")
ERROR_LOG_PATH = Path("../data/validation_errors.csv")

# The exact 19 channels we require for the final unified dataset.
REQUIRED_CHANNELS = {
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
}

TARGET_SFREQ = 200  # Hz

def validate_dataset():
    """
    Reads the master metadata file and validates each EEG recording.
    --- Implements a more robust, flexible channel validation
    that checks for substrings (e.g., 'FP1' in 'FP1-AVG') to handle
    different naming conventions across datasets. ---
    """
    print("--- Starting Phase 0, Step 4: Data Validation ---")
    if not METADATA_PATH.exists():
        print(f"\n[FATAL] Metadata file not found at {METADATA_PATH}.")
        print("Please run 'create_metadata.py' successfully before running validation.")
        sys.exit(1)

    metadata_df = pd.read_csv(METADATA_PATH)
    
    success_count = 0
    error_records = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Validating files"):
        file_path = Path(row['harmonized_file_path'])
        subject_id = row['subject_id']
        error_messages = []

        if not file_path.exists():
            error_messages.append("File not found at the specified path.")
        else:
            try:
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose='CRITICAL')

                if int(raw.info['sfreq']) != TARGET_SFREQ:
                    msg = f"Sampling rate mismatch: Expected {TARGET_SFREQ}, found {int(raw.info['sfreq'])}."
                    error_messages.append(msg)
                
                # --- ROBUST CHANNEL CHECKING ---
                available_channels = {ch for ch in raw.ch_names}
                missing_channels = set()
                
                for required_ch in REQUIRED_CHANNELS:
                    # Check if any available channel name is same as required channel name.
                    found = any(required_ch in actual_ch for actual_ch in available_channels)
                    if not found:
                        missing_channels.add(required_ch)
                
                if missing_channels:
                    msg = f"Missing required channels: {', '.join(sorted(list(missing_channels)))}."
                    error_messages.append(msg)

            except Exception as e:
                error_messages.append(f"MNE failed to read file. Error: {str(e)}")
        
        if not error_messages:
            success_count += 1
        else:
            error_records.append({
                'subject_id': subject_id,
                'file_path': str(file_path),
                'errors': "; ".join(error_messages)
            })

    print("\n--- Validation Complete ---")
    print(f"Successfully validated: {success_count} / {len(metadata_df)} files.")
    
    if error_records:
        print(f"Found {len(error_records)} files with issues.")
        error_df = pd.DataFrame(error_records)
        error_df.to_csv(ERROR_LOG_PATH, index=False)
        print(f"An error log has been saved to: {ERROR_LOG_PATH}")
        print("Please review the error log. If issues persist, the files may be fundamentally missing channels.")
    else:
        print("\n✅ All files passed the robust validation! Phase 0 is officially complete.")

if __name__ == '__main__':
    validate_dataset()