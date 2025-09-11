import mne
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
from mne.io import Raw
from mne_icalabel import label_components
import matplotlib.pyplot as plt

# --- Configuration ---
HARMONIZED_METADATA_PATH = Path("../data/harmonized_metadata.csv")
OUTPUT_DIR = Path("../data/processed/preprocessed_epochs")
OUTPUT_METADATA_PATH = Path("../data/preprocessed_metadata.csv")
SANITY_CHECK_DIR = Path("../results/figures/ica_sanity_checks")

# --- Preprocessing Parameters ---
BANDPASS_FREQ = (1.0, 45.0)
NOTCH_FREQ = 50.0
EPOCH_DURATION = 2.0
EPOCH_OVERLAP = 0.0
ICA_N_COMPONENTS = 15
ICA_RANDOM_STATE = 42
REJECT_CRITERIA = dict(eeg=100e-6)  # 100 µV

# --- Sanity Check Configuration ---
SAVE_SANITY_CHECK_PLOTS = True
N_SUBJECTS_TO_PLOT = 5

def preprocess_subject(file_path: Path) -> tuple[mne.Epochs | None, mne.preprocessing.ICA | None, dict | None]:
    """Applies the full preprocessing pipeline to a single subject's harmonized data."""
    try:
        # 1. Load Data
        raw: Raw = mne.io.read_raw_fif(file_path, preload=True, verbose='WARNING')

        # Set the standard 10-20 montage for electrode positions ---
        # This is required for topographic ICA component labeling.
        raw.set_montage('standard_1020', on_missing='raise', verbose='WARNING')

        # 2. Filtering
        raw.filter(l_freq=BANDPASS_FREQ[0], h_freq=BANDPASS_FREQ[1], fir_design='firwin', verbose='WARNING')
        raw.notch_filter(freqs=NOTCH_FREQ, verbose='WARNING')

        # 3. Re-referencing
        raw.set_eeg_reference('average', projection=False, verbose='WARNING')

        # 4. Fit ICA on continuous data
        ica = mne.preprocessing.ICA(
            n_components=ICA_N_COMPONENTS,
            method='fastica',
            random_state=ICA_RANDOM_STATE
        )
        ica.fit(raw)

        # 5. Automatically label ICA components
        component_labels = label_components(raw, ica, method='iclabel')
        labels = component_labels['labels']
        exclude_indices = [idx for idx, label in enumerate(labels) if label not in ['brain', 'other']]
        
        if not exclude_indices:
            print(f"\n[INFO] No artifactual ICA components found for {file_path.stem}. No components will be removed.")
        
        ica.exclude = exclude_indices

        # 6. Create Epochs
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=EPOCH_DURATION,
            overlap=EPOCH_OVERLAP,
            preload=True,
            verbose='WARNING'
        )

        # 7. Apply ICA to remove artifacts
        ica.apply(epochs, verbose='WARNING')

        # 8. Data Quality Control: Reject remaining bad epochs
        epochs.drop_bad(reject=REJECT_CRITERIA, verbose='WARNING') # type: ignore
        
        return epochs, ica, component_labels

    except Exception as e:
        print(f"\n[ERROR] Could not process {file_path}. Reason: {e}")
        return None, None, None

def main():
    """Main orchestration function to run the preprocessing pipeline."""
    print("--- Starting Phase 1: Unified Preprocessing Pipeline (v2) ---")
    if not HARMONIZED_METADATA_PATH.exists():
        print(f"[FATAL] Harmonized metadata not found. Exiting.")
        sys.exit(1)

    metadata_df = pd.read_csv(HARMONIZED_METADATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_SANITY_CHECK_PLOTS:
        SANITY_CHECK_DIR.mkdir(parents=True, exist_ok=True)

    preprocessed_records = []

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Preprocessing files"):
        file_path = Path(row['harmonized_file_path'])
        
        epochs, ica, labels = preprocess_subject(file_path)

        if epochs is not None and len(epochs) > 0:
            # Save the clean epochs
            source_dataset = row['original_dataset_source']
            diagnosis = row['diagnosis']
            original_subject_id = row['original_subject_id']
            # Sanitize the original subject ID for use in a filename (e.g., replace spaces)
            sanitized_subject_id = str(original_subject_id).replace(" ", "_")
            sanitized_subject_id = str(original_subject_id).replace("-", "_")
            output_filename = f"{source_dataset}_preprocessed_epochs_{diagnosis}_{sanitized_subject_id}_epo.fif"

            output_filepath = OUTPUT_DIR / output_filename
            epochs.save(output_filepath, overwrite=True, verbose='WARNING')

            # Create a record for the new metadata file
            new_record = row.to_dict()
            new_record['preprocessed_epo_path'] = str(output_filepath.resolve())
            new_record['n_clean_epochs'] = len(epochs)
            preprocessed_records.append(new_record)

            # Perform sanity check plotting for the first few subjects
            if SAVE_SANITY_CHECK_PLOTS and idx < N_SUBJECTS_TO_PLOT: # type: ignore
                try:
                    fig_components = ica.plot_components(show=False) # type: ignore
                    fig_path = SANITY_CHECK_DIR / f"{row['subject_id']}_ica_components.png"
                    fig_components.savefig(fig_path) # type: ignore
                    plt.close(fig_components) # type: ignore
                    
                    if ica.exclude: # type: ignore
                        figs_properties = ica.plot_properties(epochs, picks=ica.exclude, show=False) # type: ignore
                        for i, fig in enumerate(figs_properties):
                            comp_idx = ica.exclude[i] # type: ignore
                            fig.savefig(SANITY_CHECK_DIR / f"{row['subject_id']}_ica_properties_comp_{comp_idx}.png")
                        plt.close('all')

                except Exception as e:
                    print(f"\n[WARNING] Could not save sanity check plot for {row['subject_id']}. Reason: {e}")
        else:
            print(f"\n[INFO] No clean epochs remaining for {row['subject_id']} after preprocessing. Skipping.")

    # Save the final metadata file
    if preprocessed_records:
        preprocessed_df = pd.DataFrame(preprocessed_records)
        preprocessed_df.to_csv(OUTPUT_METADATA_PATH, index=False)
        print("\n--- Preprocessing Complete ---")
        print(f"Successfully processed and saved {len(preprocessed_records)} files.")
        print(f"✅ New metadata for the next phase saved to: {OUTPUT_METADATA_PATH}")
    else:
        print("\n--- Preprocessing Failed ---")

if __name__ == '__main__':
    main()
