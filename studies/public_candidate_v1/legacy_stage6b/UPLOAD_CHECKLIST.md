# GitHub and Zenodo release checklist

## Before making the repository public

- [x] Add the Apache-2.0 / CC BY 4.0 license mapping and full license texts.
- [x] Add author email and ORCID to `CITATION.cff`.
- [x] Confirm the author and affiliation spelling.
- [x] Run `python -B scripts/analyze_stage6b.py` and compare the printed report
      with `results/stage6b/stage6b_heldout_aggregate/REPORT.md`.
- [x] Run the fake-data smoke test from `README.md`.
- [x] Run `powershell -File scripts/generate_checksums.ps1`.
- [ ] Check `git status` before committing; CIFAR-10 and smoke-test outputs
      should remain ignored.

## Suggested first release

1. Create a GitHub repository and upload the contents of this `github` folder
   as the repository root.
2. Commit the files and push the default branch.
3. Connect the GitHub repository to Zenodo and enable archiving.
4. Create a GitHub release tagged `v1.0.0`.
5. Let Zenodo archive the release and mint the DOI.
6. Add the GitHub URL, Zenodo concept DOI, and version DOI to `README.md` and
   `CITATION.cff`, then commit that metadata update.
7. Update the manuscript with the public repository URL and Zenodo DOI;
   author email, ORCID, funding, and competing-interest declarations are already complete.
8. Submit the final manuscript to *Neural Networks*.
9. Immediately after submission, post the exact submitted manuscript to arXiv
   and add it to the GitHub repository.
10. Use the concept DOI in the manuscript when a DOI that follows future
   releases is preferred; preserve the version DOI for the exact archived
   release.

## Journal submission files kept outside this public package

The manuscript DOCX/PDF, cover letter, submission-readiness notes, and
journal-upload administrative files were intentionally not copied into the
first public release. The manuscript will be added after the repository DOI is available and the
repository URL/DOI have been inserted into the final submission version.
