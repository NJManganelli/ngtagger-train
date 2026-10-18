"""The census formats the loader reads (tp_findability.READ_FORMATS): the current one (36), 35 (WP-D's) and 34,
the last census built from canon_v3. load() records the format it read and refuses any other; a comparison or aggregation of
censuses (one_format) refuses a mix of formats unless it is marked cross-format."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"))
import tp_findability as TF  # noqa: E402

# a format-34 census on canon_v3 (WP-D0b run OFF34)
F34 = Path("/Volumes/WDMac/smartpixels_scratch/wpd0b/census/OFF34")


def test_read_formats():
    assert TF.READ_FORMATS == (34, 35, 36) and TF.FORMAT_VERSION == 36


def test_other_format_refused(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": 33}))
    with pytest.raises(SystemExit, match="census format 33; this code reads 34, 35, 36 only"):
        TF.load(str(tmp_path), verbose=False)


def test_one_format():
    assert TF.one_format({"a": {"format": 36}, "b": {"format": 36}}) == (36,)
    with pytest.raises(SystemExit, match=r"census formats differ \(wpd: 35, wpdisc: 36\)"):
        TF.one_format({"wpd": {"format": 35}, "wpdisc": {"format": 36}})
    mix = {"canon_v3": {"format": 34}, "canon_v4": {"format": 35}}
    with pytest.raises(SystemExit, match=r"census formats differ \(canon_v3: 34, canon_v4: 35\)"):
        TF.one_format(mix)
    assert TF.one_format(mix, cross_format=True) == (34, 35)


@pytest.mark.skipif(not F34.exists(), reason="WDMac wpd0b censuses not mounted")
def test_reads_format_34_and_records_it():
    C = TF.load(TF.census_dir(str(F34)), nev=4, with_tracks=True, verbose=False)
    assert C["format"] == 34
    assert C["n_events"] >= 4 and len(C["key"]) and C["tracks"]
    with pytest.raises(SystemExit, match="census formats differ"):
        TF.one_format({"format 34": C, "format 35": {"format": 35}})


def test_hit_key_every_kind():
    for kind in ("", "dr", "m", "s", "sm", "o", "om"):
        assert TF.hit_key(f"hit_{kind}12") == (kind, 12)
    with pytest.raises(SystemExit, match="unknown shard key"):
        TF.hit_key("hit_12x")
