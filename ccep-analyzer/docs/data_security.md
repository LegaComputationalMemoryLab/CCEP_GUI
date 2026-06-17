# Data security and PHI guidance

Do not commit or upload any of the following to a public GitHub repository:

- EDF/BDF/FIF/NWB recordings.
- Stimulation logs containing dates, medical record numbers, initials, or clinical event metadata.
- Electrode coordinate CSV files from actual patients.
- MRI, CT, DICOM, NIfTI, FreeSurfer, or surface files generated from identifiable clinical imaging.
- Screenshots or reports that contain patient identifiers, dates, clinical text, or rare implantation patterns.
- PDF/HTML reports generated from clinical data.

For manuscript submission, patient-level recordings, stimulation logs, and coordinate files should remain unavailable publicly if they contain clinical data or potentially identifiable information. Use institutional review and data-use agreements for qualified external requests.

Recommended release practice:

1. Publish code only.
2. Include synthetic coordinate examples only.
3. Keep the repository private until PI/legal review is complete.
4. Scan the repository before public release:

```bash
git status --ignored
git ls-files
git log --stat --all
```

5. If PHI was ever committed, do not merely delete it in a later commit. Rewrite history or create a clean repository, then have the repository reviewed before public release.
