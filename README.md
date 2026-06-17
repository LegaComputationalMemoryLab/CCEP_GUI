# Python-Based CCEP Analyzer

A Python graphical workflow for presurgical cortico-cortical evoked potential (CCEP) review in drug-resistant epilepsy. The analyzer consolidates EDF import, stimulation-event parsing, configurable preprocessing, event-related potential summaries, root-mean-square summaries, low-gamma summaries, region-of-interest visualization, 3D electrode review, static export, interactive HTML export, and PDF reporting.

This repository is intended to accompany the manuscript:

> Pruitt T, Zafarmandi Ardabili S, Knox C, Podkorytova I, Gibson W, Wu C, Davila CE, Lega B. *A Python-Based Analyzer for Presurgical Cortico-Cortical Evoked Potential Mapping in Drug-Resistant Epilepsy.* Submitted to Clinical Neurophysiology.

## Important clinical and data-use notice

This software is a research and clinical-review support tool. It is not a standalone diagnostic device and must not be used as the sole basis for surgical or neuromodulation decisions. Outputs require review by qualified clinical teams and must be interpreted with seizure semiology, intracranial EEG, imaging, functional mapping, safety testing, and standard presurgical conference data.

Do not commit patient-level EDF files, stimulation logs, coordinate files, MRI/CT/DICOM/NIfTI data, exported reports, screenshots containing identifiers, or any protected health information (PHI) to this repository. The `.gitignore` file is intentionally conservative.

## Features

- EDF import using MNE-Python.
- Stimulation annotation parsing and event grouping by stimulation pair.
- Configurable notch and band-pass filtering.
- Epoch extraction with baseline correction.
- ERP, RMS, and low-gamma response summaries.
- Early and late response-window review.
- Coordinate CSV import with contact-level anatomical labels.
- Region-of-interest filtering and top-responding ROI review.
- 3D pial, white-matter, and inflated surface visualization.
- Interactive HTML viewer export.
- PDF report export with electrode-placement and activation views.
- Session-level workflow timing export to `workflow_timing_metrics.csv`.

## Repository layout

```text
.
├── src/ieeg_ccep_analyzer/app.py       # Full analyzer GUI script
├── run_analyzer.py                     # Convenience launcher after editable install
├── examples/synthetic_coordinates.csv  # Synthetic coordinate table only, no patient data
├── docs/coordinate_file_format.md      # Coordinate CSV requirements
├── docs/data_security.md               # PHI/data-handling guidance
├── docs/release_checklist.md           # Release and manuscript-code checklist
├── docs/zenodo_release_steps.md        # How to archive a GitHub release with a DOI
├── pyproject.toml                      # Package metadata and dependencies
├── requirements.txt                    # pip dependency list
└── environment.yml                     # conda environment
```

## Installation

### Option A: conda

```bash
conda env create -f environment.yml
conda activate ccep-analyzer
pip install -e .
```

### Option B: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Linux users may need to install Tkinter separately, for example `sudo apt-get install python3-tk`.

## Running the analyzer

After installation:

```bash
ieeg-ccep-analyzer
```

Or from the repository root:

```bash
python run_analyzer.py
```

## Minimal workflow

1. Launch the analyzer.
2. Select an EDF file.
3. Load the coordinate CSV file.
4. Choose the stimulation pair and analysis mode: ERP, RMS, or Gamma.
5. Run analysis.
6. Review waveform and 3D outputs.
7. Export the interactive HTML viewer or PDF report as needed.

## Coordinate CSV format

The coordinate table is generated outside the analyzer from the local electrode-localization workflow. Required columns:

| Column | Required | Units | Description |
|---|---:|---|---|
| `ElecNumber` | Yes | integer | Contact order matched to EDF channel order. |
| `X` | Yes | millimeters | Contact x-coordinate in the chosen coordinate frame. |
| `Y` | Yes | millimeters | Contact y-coordinate in the chosen coordinate frame. |
| `Z` | Yes | millimeters | Contact z-coordinate in the chosen coordinate frame. |
| `Regions` | No | text | Region label used for ROI filtering and summaries. |

See `examples/synthetic_coordinates.csv` for a non-patient example.

## Outputs

Typical outputs include:

- Static figures: PNG, PDF, SVG.
- Interactive HTML report.
- PDF report.
- Channel-level and trial-level metrics.
- Workflow timing CSV.

Do not upload generated outputs to GitHub if they contain clinical information or potentially identifying anatomy/metadata.

## Testing

```bash
python -m py_compile src/ieeg_ccep_analyzer/app.py
python -m pytest
```

The included automated test is intentionally minimal because the GUI requires clinical data and an interactive display for full validation.

## Versioning and citation

Use semantic versioning for releases. The manuscript-submission release should be tagged, for example:

```bash
git tag -a v0.1.0 -m "Initial manuscript release"
git push origin v0.1.0
```

After creating the GitHub release, archive it with Zenodo and add the DOI to `CITATION.cff`, the GitHub release notes, and the manuscript Code availability section.

## License

A license must be confirmed with the principal investigator and institutional technology-transfer/legal office before public release. Two license templates are provided in `docs/license_options.md`. Replace the placeholder `LICENSE` file with the approved license before making the repository public.

## Contact

Corresponding author: Bradley Lega, MD  
Department of Neurological Surgery, University of Texas Southwestern Medical Center  
Email: Bradley.Lega@UTSouthwestern.edu
