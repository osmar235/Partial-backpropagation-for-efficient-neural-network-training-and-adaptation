# Archiving this repository with Zenodo

## Completed archive

Release `v1.0.1`, commit `be13a61d9ffe9ba657a9a549dfaa278aa8e4c622`, is publicly archived at [10.5281/zenodo.24268905](https://doi.org/10.5281/zenodo.24268905). The public Zenodo API confirms the title, author, version, release link and uploaded source ZIP. Its concept DOI is [10.5281/zenodo.24268904](https://doi.org/10.5281/zenodo.24268904). Cite the version DOI for the manuscript.

The original failed `v1.0.0` release and successful `v1.0.1` tag are preserved. Adding the DOI to `main` does not require another release. For future scientific/source changes, update the version metadata and create a new release; do not put the existing version DOI into `.zenodo.json` as the DOI of a new deposit.

## Historical setup and troubleshooting

The instructions and pending-status statements below document the setup before successful archival.

1. In Zenodo, connect the GitHub account and enable this repository.
2. Create a GitHub release from the reviewed publication commit after the connection is active. Use a version tag such as `v1.0.0` if that tag is still unused.
3. Confirm that Zenodo has archived the release and review the title, author, licenses and included files.
4. Copy the version-specific DOI into the manuscript and repository citation metadata. The version DOI identifies the exact archived package; the concept DOI refers to the evolving record.
5. Commit metadata updates separately and make a later release if the archived contents need to change. Do not silently move an existing release tag to a different commit.

No DOI is asserted in this pre-submission package. The manuscript's GitHub availability wording should be used only after public access to the uploaded files is verified.

## Metadata compatibility update for v1.0.1

The v1.0.0 GitHub release was published, but the owner reported that Zenodo marked its import as failed. Its valid CFF license array matches a documented Zenodo ingestion incompatibility; the exact cause remains subject to the failed release's Errors detail.

The next release includes `.zenodo.json`, which Zenodo reads in preference to `CITATION.cff`. This avoids the CFF array-conversion path while retaining the complete multi-license CFF metadata for GitHub. The Zenodo software license field is a scalar Apache-2.0 identifier; its description explicitly retains CC BY 4.0 for research artifacts and all third-party terms. No files are relicensed.

Publish a new `v1.0.1` release from the corrected `main` commit after reviewing the error. Preserve the existing `v1.0.0` tag. Merely editing `main` does not change the old release snapshot or retry its import. If the new import also fails, open its Errors detail before another attempt. No archival DOI is claimed until Zenodo confirms success.

Sources: [Zenodo metadata precedence](https://help.zenodo.org/docs/github/describe-software/zenodo-json/), [failed-release troubleshooting](https://help.zenodo.org/docs/github/archive-software/github-upload/), [CFF license-array issue](https://github.com/zenodo/zenodo/issues/2515).
