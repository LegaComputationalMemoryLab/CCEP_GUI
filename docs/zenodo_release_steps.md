# Zenodo release DOI steps

1. Create or log in to a Zenodo account.
2. Link Zenodo to the lab GitHub organization.
3. Enable archiving for the repository.
4. In GitHub, create a release such as `v0.1.0`.
5. Zenodo will archive the release and mint a DOI.
6. Copy the DOI into:
   - `CITATION.cff`
   - the GitHub release notes
   - the manuscript Code availability section
   - the Clinical Neurophysiology submission metadata, if requested

Suggested manuscript wording after release:

> Code availability: The source code for the Python-based CCEP analyzer is available at [GitHub repository URL] under the [approved license] license. The archived release corresponding to this manuscript is available at [Zenodo DOI]. Patient-level recordings, stimulation logs, and coordinate files are not included because they contain clinical data and potentially identifiable information.
