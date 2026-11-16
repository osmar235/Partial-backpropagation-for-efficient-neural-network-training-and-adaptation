# Manuscript-support files

This directory intentionally omits the manuscript DOCX and PDF from GitHub /
Zenodo release v1.0.0. The first public release establishes the repository and
archival DOI before the final manuscript is prepared.

Included files:

- `figures/`: publication-resolution Figures 1-3;
- `rebuild_submission_figures.py`: reproducible figure-generation source;
- `references_revised.bib`: bibliography used by the submission draft.

Regenerate the standalone figures with:

```powershell
python -B manuscript/rebuild_submission_figures.py
```

After Zenodo mints the DOI, the final manuscript will be updated with the
repository URL and persistent DOI. Author email, ORCID, funding, and competing-
interest declarations are already complete. The exact submitted manuscript will then be posted to
arXiv and added to this repository immediately after journal submission.

