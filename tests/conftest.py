"""Shared test fixtures."""
import os

import pytest


@pytest.fixture
def jet_nano_path():
    """Path to a JET-tagger nano for real-data smoke tests, or a clean skip.

    Skips when NGTAGGER_TEST_NANO is unset, and — importantly — also when it
    points at a nano that lacks the jet-constituent link table (e.g. a
    track-tagger nano like the SmartPixels refit files). The jet pipeline
    (prepare_dataset) needs L1SC4NGJetCands; pointing it at a track-only nano
    would otherwise hard-fail with a confusing awkward FieldNotFoundError
    instead of skipping.
    """
    path = os.environ.get("NGTAGGER_TEST_NANO")
    if not path:
        pytest.skip("set NGTAGGER_TEST_NANO to a jet nano to run this real-data smoke")
    import uproot

    with uproot.open(path) as f:
        keys = list(f["Events"].keys()) if "Events" in f else []
    if not any("L1SC4NGJetCands" in k for k in keys):
        pytest.skip(
            f"NGTAGGER_TEST_NANO={os.path.basename(path)} has no jet-tagger tables "
            "(L1SC4NGJetCands); this jet-pipeline test needs a jet nano, not a "
            "track-only nano — point a separate jet nano at it to exercise this path"
        )
    return path
