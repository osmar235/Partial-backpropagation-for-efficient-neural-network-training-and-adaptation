# Oxford download timeout correction

The v1 helper imposed curl --max-time 3600. The user's four full transfers were
still progressing when that one-hour deadline terminated them. Each subsequent
resume failed because the server did not serve byte ranges. This was a downloader
configuration error, not an experiment failure. No baseline training has started.

Run this replacement command from CMD:

```
cd /d C:\Users\osmar\projects\Projeto_AI_Fronteira
set "PYTHON=C:\Users\osmar\projects\.venv\Scripts\python.exe"
call scripts\run_pets_baseline_retry_v2.cmd
```

V2 explicitly sets overall transfer time to unlimited and ignores curlrc defaults.
It retains the 20-second connection timeout and a 90-second low-speed timeout
(less than 1 KB/s), certificate verification, checksum verification and safe
extraction. A transfer can continue beyond one hour while making progress.

Existing partial archives are retained. The currently incomplete image archive
will be tried for resume; if the server still rejects ranges, it must start again.
The already verified annotation archive is reused. At the observed rates, allow
roughly 75–90 minutes for a fresh image transfer, plus extraction/cache creation
and training; throughput and connection availability can change. Keep the machine
awake and the CMD window open. Do not run the older retry launcher alongside v2.

All original experiment code, seeds, thresholds and data definitions are unchanged.
V1 is preserved for provenance. V2 imports its tested checksum/extraction/retry
logic and changes only the curl command. Seven tests pass, including a simulated
transfer longer than an hour and all five previous recovery tests. This does not
claim that the full real image archive has already downloaded.
