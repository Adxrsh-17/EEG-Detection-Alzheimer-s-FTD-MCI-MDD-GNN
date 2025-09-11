import mne
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import sys

# --- Configuration ---
# This script assumes it is located in the 'code/' directory.
# Paths are relative to the script's location.
MASTER_METADATA_PATH = Path("../data/master_metadata.csv")
RAW_DATA_BASE_PATH = Path("../data/raw") # For CAUEEG event files
OUTPUT_DIR = Path("../data/processed/harmonized")
OUTPUT_METADATA_PATH = Path("../data/harmonized_metadata.csv")

# --- Harmonization Parameters ---
TARGET_SAMPLING_RATE = 200  # Hz
STANDARD_19_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

def rename_channels(raw):
    """
    Renames channels to a standard 19-channel montage.
    This function handles various naming conventions (e.g., 'FP1-AVG', 'EEG FP1').
    """
    rename_mapping = {}
    ch_names_upper = [ch.upper() for ch in raw.ch_names]
    
    for standard_ch in STANDARD_19_CHANNELS:
        for i, actual_ch_upper in enumerate(ch_names_upper):
            if standard_ch.upper() in actual_ch_upper:
                rename_mapping[raw.ch_names[i]] = standard_ch
                break
    
    raw.rename_channels(rename_mapping)
    return raw

def extract_eyes_closed_caueeg(raw, subject_id: str):
    """
    Extracts only the 'Eyes Closed' segments for a CAUEEG recording.
    --- V5 Update: Makes the end-time calculation robust against
    event markers that exceed the raw data's duration. ---
    """
    event_json_path = RAW_DATA_BASE_PATH / "caueeg" / "event" / f"{subject_id}.json"
    if not event_json_path.exists():
        print(f"  [Warning] Event file not found for CAUEEG subject {subject_id}. Using full recording.")
        return raw

    with open(event_json_path, 'r') as f:
        events = json.load(f)

    ec_segments = []
    sfreq = raw.info['sfreq']
    max_time = raw.times[-1]  # Get the maximum possible time once

    for i, event in enumerate(events):
        timestamp, description = event
        if "eyes closed" in description.lower():
            start_time_sec = timestamp / sfreq
            
            # Robustly determine the end time of the segment ---
            end_time_sec = max_time # Default to the end of the file
            if i + 1 < len(events):
                end_time_sec = events[i+1][0] / sfreq
            
            # Ensure the end time does not exceed the recording's actual length ---
            end_time_sec = min(end_time_sec, max_time)
            
            duration = end_time_sec - start_time_sec
            # Use a small epsilon to avoid floating point precision issues
            if duration > 1e-5: 
                ec_segments.append(raw.copy().crop(tmin=start_time_sec, tmax=end_time_sec))

    if not ec_segments:
        # Return an empty Raw object if no valid segments are found
        print(f"  [Warning] No valid 'Eyes Closed' segments found for {subject_id}. Returning empty recording.")
        return raw.crop(tmax=0)
    
    return mne.concatenate_raws(ec_segments)

def main():
    """
    Main orchestration function to run the harmonization pipeline.
    """
    print("--- Starting Phase 1, Step 1: Data Harmonization (v2) ---")
    if not MASTER_METADATA_PATH.exists():
        print(f"[FATAL] Master metadata file not found at {MASTER_METADATA_PATH}. Exiting.")
        sys.exit(1)
        
    metadata_df = pd.read_csv(MASTER_METADATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    harmonized_records = []
    
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Harmonizing files"):
        try:
            file_path = Path(row['file_path'])
            source_dataset = row['original_dataset_source']
            original_subject_id = row['original_subject_id']

            # 1. Load the raw EDF file
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='WARNING')

            # This prevents crashes for files with corrupted/future dates.
            raw.set_meas_date(None)
            
            # 2. Rename channels to the standard 10-20 system
            raw = rename_channels(raw)

            # 3. Select only the 19 standard channels in a consistent order
            raw.pick_channels(STANDARD_19_CHANNELS, ordered=True)
            
            # 4. Extract "Eyes Closed" condition
            # --- MODIFICATION START ---
            if source_dataset == 'caueeg':
                # For CAUEEG, we parse event files to extract EC segments
                raw = extract_eyes_closed_caueeg(raw, original_subject_id)
            elif source_dataset in ['ds004504', 'figshare_mdd']:
                # For these datasets, the metadata creation step already selected 
                # the specific eyes-closed files, so no further extraction is needed.
                pass 
            # --- MODIFICATION END ---
            
            # 5. Resample the data to the target frequency
            if raw.info['sfreq'] != TARGET_SAMPLING_RATE: # type: ignore
                raw.resample(TARGET_SAMPLING_RATE, verbose='WARNING') # type: ignore

            # 6. Save the harmonized data to a .fif file
            diagnosis = row['diagnosis']
            # Sanitize the original subject ID for use in a filename (e.g., replace spaces)
            sanitized_subject_id = original_subject_id.replace(" ", "_")
            sanitized_subject_id = original_subject_id.replace("-", "_")
            output_filename = f"{source_dataset}_harmonized_{diagnosis}_{sanitized_subject_id}_eeg.fif"
            output_filepath = OUTPUT_DIR / output_filename
            raw.save(output_filepath, overwrite=True, verbose='WARNING') # type: ignore

            # 7. Add a record for the new metadata file
            new_record = row.to_dict()
            new_record['harmonized_file_path'] = str(output_filepath.resolve())
            new_record['sampling_rate'] = int(raw.info['sfreq']) # type: ignore
            harmonized_records.append(new_record)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process file {file_path}. Subject ID: {row['subject_id']}. Error: {e}") # type: ignore

    # 8. Save the new metadata file for the next phase
    if harmonized_records:
        harmonized_df = pd.DataFrame(harmonized_records)
        harmonized_df.to_csv(OUTPUT_METADATA_PATH, index=False)
        print("\n--- Harmonization Complete ---")
        print(f"Successfully processed and saved {len(harmonized_records)} files to: {OUTPUT_DIR}")
        print(f"✅ New metadata for the next phase saved to: {OUTPUT_METADATA_PATH}")
    else:
        print("\n--- Harmonization Failed ---")
        print("No files were processed successfully.")

if __name__ == '__main__':
    main()