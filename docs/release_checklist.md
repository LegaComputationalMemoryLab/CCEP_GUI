# Release checklist for manuscript submission

## Before GitHub release

- [ ] Confirm repository name and owner with the lab.
- [ ] Confirm license with PI and institutional technology-transfer/legal office.
- [ ] Confirm no patient data are present.
- [ ] Confirm no clinical dates, initials, accession numbers, MRNs, or screenshots are present.
- [ ] Confirm the script compiles: `python -m py_compile src/ieeg_ccep_analyzer/app.py`.
- [ ] Confirm README installation instructions work on at least one clean environment.
- [ ] Confirm synthetic example files contain no real patient information.
- [ ] Create version tag, for example `v0.1.0`.
- [ ] Create GitHub release.
- [ ] Archive the release with Zenodo.
- [ ] Update `CITATION.cff` with repository URL, license, and DOI.
- [ ] Update manuscript Code availability section.

## Before journal submission

- [ ] Replace `GitHub Link:` placeholder in the manuscript.
- [ ] Insert the archived release DOI if available.
- [ ] Ensure the code availability statement matches the repository license and version.
- [ ] Ensure the data availability statement says patient data are restricted.
- [ ] Ensure AI declaration is accurate.
