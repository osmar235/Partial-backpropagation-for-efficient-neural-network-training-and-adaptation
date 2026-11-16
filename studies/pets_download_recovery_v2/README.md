# Oxford Pet download recovery v2

Start with README_PETS_DOWNLOAD_RECOVERY_v2.md and the retry_v2 launcher. The v1
helper had a one-hour overall deadline; v2 disables that deadline and curlrc
defaults while retaining connection/stall timeouts, checksums and safe extraction.
All original scientific sources remain unchanged. This package includes the
original helper imported by v2 and all baseline source dependencies.

Seven focused tests passed, including simulated long-transfer completion and all
five previous recovery tests. The complete real image download has not yet been
verified; the smaller official annotations archive was previously verified.
No new model training, evaluation, threshold changes or external release.
