import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import uuid

# --- Configuration ---
# This script assumes it is located in the 'code/' directory.
# The paths are relative to the script's location.
BASE_DATA_PATH = Path("../data/raw")
OUTPUT_METADATA_PATH = Path("../data/master_metadata.csv")

def process_openneuro_ds004504():
    """
    Processes the OpenNeuro dataset (ds004504) from your specific file structure.
    It reads the main participants.tsv and locates the corresponding EDF file.
    --- V2 Update: Correctly reads the 'Group' column instead of 'diagnosis'
    and maps the group codes (A, F, C) to the final labels. Also uses
    correct capitalization for 'Age' and 'Gender' columns. ---
    """
    dataset_path = BASE_DATA_PATH / "ds004504"
    participants_path = dataset_path / "participants.tsv"
    records = []
    
    print("Processing OpenNeuro dataset (ds004504)...")
    if not participants_path.exists():
        print(f"  [ERROR] participants.tsv not found at {participants_path}")
        return []

    participants_df = pd.read_csv(participants_path, sep='\t')

    for _, row in tqdm(participants_df.iterrows(), total=len(participants_df), desc="  -> Scanning subjects"):
        subject_id = row['participant_id']
        
        group_code = row['Group']
        
        label = None
        if group_code == 'C':
            label = 'CN'
        elif group_code == 'A':
            label = 'AD'
        elif group_code == 'F':
            label = 'FTD'
        else:
            continue

        eeg_file_path = dataset_path / "raw_data-edf_converted" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.edf"
        
        if eeg_file_path.exists():
            records.append({
                'original_subject_id': subject_id,
                'file_path': str(eeg_file_path.resolve()),
                'diagnosis': label,
                'original_dataset_source': 'ds004504',
                'age': row.get('Age', None),
                'sex': row.get('Gender', None),
                'sampling_rate': 500
            })
        else:
            print(f"  [WARNING] EEG file not found for {subject_id} at {eeg_file_path}")
            
    return records

def process_caueeg():
    """
    Processes the CAUEEG dataset from your specific file structure.
    --- V3 Update: Reads the master 'annotation.json' file to get diagnosis info.
    It now iterates through EDF files and uses the annotation file as a lookup table. ---
    """
    dataset_path = BASE_DATA_PATH / "caueeg"
    # The diagnosis information is in a single master file.
    annotation_path = dataset_path / "annotation.json"
    signal_path = dataset_path / "signal" / "edf"
    records = []

    print("Processing CAUEEG dataset...")
    if not annotation_path.exists() or not signal_path.exists():
        print(f"  [ERROR] CAUEEG annotation.json or signal directory not found.")
        return []

    # 1. Load the master annotation file once.
    with open(annotation_path, 'r') as f:
        master_meta = json.load(f)

    # 2. Create a lookup dictionary for fast access using the subject's serial number.
    subject_lookup = {item['serial']: item for item in master_meta['data']}

    # 3. Iterate through the actual EEG files, not the event files.
    edf_files = list(signal_path.glob("*.edf"))

    for edf_file in tqdm(edf_files, desc="  -> Scanning subjects"):
        original_subject_id = edf_file.stem
        
        # 4. Find the subject's metadata in our lookup table.
        subject_data = subject_lookup.get(original_subject_id)
        
        if not subject_data:
            print(f"  [WARNING] Metadata for {original_subject_id} not found in annotation.json. Skipping.")
            continue

        # 5. Get symptoms from the correct key 'symptom' (not 'symptom_tags')
        symptoms = subject_data.get('symptom', [])
        label = None

        if any(tag in symptoms for tag in ['dementia', 'ad', 'load']):
            label = 'AD'
        elif any(tag in symptoms for tag in ['mci', 'mci_amnestic', 'mci_amnestic_rf']):
            label = 'MCI'
        elif any(tag in symptoms for tag in ['normal', 'cb_normal']):
            label = 'CN'
        
        if label:
            records.append({
                'original_subject_id': original_subject_id,
                'file_path': str(edf_file.resolve()),
                'diagnosis': label,
                'original_dataset_source': 'caueeg',
                'age': subject_data.get('age', None),
                'sex': subject_data.get('gender', None),
                'sampling_rate': 200
            })
            
    return records

def process_figshare_mdd():
    """
    Processes the Figshare MDD dataset from your specific folder structure.
    It scans both 'H Subjects' and 'MDD Subjects' folders for Eyes-Closed (EC) data.
    """
    dataset_path = BASE_DATA_PATH / "figshare_mdd"
    records = []

    print("Processing Figshare MDD dataset...")
    if not dataset_path.exists():
        print(f"  [ERROR] Figshare MDD directory not found at {dataset_path}")
        return []

    edf_files = list(dataset_path.glob("* Subjects/*EC.edf"))

    for edf_file in tqdm(edf_files, desc="  -> Scanning subjects"):
        filename = edf_file.stem
        label = None

        if 'MDD' in filename:
            label = 'MDD'
        elif 'H S' in filename:
            label = 'CN'
        
        if label:
            records.append({
                'original_subject_id': filename,
                'file_path': str(edf_file.resolve()),
                'diagnosis': label,
                'original_dataset_source': 'figshare_mdd',
                'age': None,
                'sex': None,
                'sampling_rate': 256
            })

    return records

def main():
    """
    Main function to orchestrate the creation of the master metadata file.
    """
    print("--- Starting Phase 0, Step 3: Master Metadata Generation (v3) ---")
    
    all_records = []
    all_records.extend(process_openneuro_ds004504())
    all_records.extend(process_caueeg())
    all_records.extend(process_figshare_mdd())

    if not all_records:
        print("\n[FATAL] No records were generated. Please double-check your data paths and file structures inside 'data/raw/'.")
        return

    df = pd.DataFrame(all_records)
    df['subject_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    column_order = [
        'subject_id', 'original_subject_id', 'diagnosis', 'file_path', 
        'original_dataset_source', 'sampling_rate', 'age', 'sex'
    ]
    df = df[column_order]

    OUTPUT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_METADATA_PATH, index=False)

    print(f"\n--- Metadata Generation Complete ---")
    print(f"Successfully processed {len(df)} records.")
    print("\nFinal Class Distribution:")
    print(df['diagnosis'].value_counts())
    print(f"\n✅ Master metadata file saved to: {OUTPUT_METADATA_PATH}")

if __name__ == '__main__':
    main()