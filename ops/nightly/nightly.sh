#!/bin/bash
# Run by launchd (ch.smartpixels.nightly) at 03:30 local: registry snapshot, then the job-copy prune.
D=$(cd "$(dirname "$0")" && pwd)
bash "$D/registry_backup.sh"
bash "$D/prune_job_copies.sh"
