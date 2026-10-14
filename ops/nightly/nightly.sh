#!/bin/bash
# Run by launchd (ch.smartpixels.nightly) at 03:30 local: prune frozen job copies older than 30 days.
# (The registry snapshot runs after each heavy job, from ops/spheavy; see ops/registry_snapshot.sh.)
D=$(cd "$(dirname "$0")" && pwd)
bash "$D/prune_job_copies.sh"
