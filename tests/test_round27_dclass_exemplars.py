"""Round 27 (6.11.2026): the D Classes tab shows two exemplars per class
with the labelled box drawn. Pinned ids win when they are single-label
rows of their class; otherwise the largest labelled boxes the detector
agrees on fill the slots, one per photo.
"""
import json
import sqlite3

import pytest

import roadguard_api as rg


def _row(pid, cls, box, model=None, n_labels=1, stem=None):
    stem = stem or f"firc_roadcrack_{pid}"
    labels = [{"class": cls, **box}] * n_labels
    return (pid, f"{stem}_jpg.rf.{pid:032x}.jpg", n_labels, cls if n_labels == 1 else f"{cls},{cls}",
            json.dumps(labels), model)


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.execute(
        "CREATE TABLE dataset_images (pic_id INTEGER PRIMARY KEY, file_name TEXT, "
        "n_labels INTEGER, classes TEXT, labels_json TEXT, model_top1_class TEXT)"
    )
    rows = [
        _row(10, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}, model="D40"),
        _row(11, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6}, model="D40"),
        _row(12, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.6, "h": 0.6}, model="D40", stem="firc_roadcrack_11"),
        _row(13, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}, model="D40"),
        _row(14, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.9, "h": 0.9}, model="D20"),
        _row(15, "D40", {"cx": 0.5, "cy": 0.5, "w": 0.9, "h": 0.9}, model="D40", n_labels=2),
        _row(20, "D20", {"cx": 0.5, "cy": 0.5, "w": 0.9, "h": 0.9}, model="D20"),
    ]
    c.executemany("INSERT INTO dataset_images VALUES (?, ?, ?, ?, ?, ?)", rows)
    return c


def test_pinned_ids_come_first_in_order(con):
    out = rg._dclass_exemplars(con, "D40", pinned=(13, 10))
    assert [p for p, _ in out] == [13, 10]
    assert out[0][1] == {"cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}


def test_fallback_ranks_by_box_area_with_detector_agreement(con):
    out = rg._dclass_exemplars(con, "D40", pinned=())
    # 14 (detector says D20) and 15 (two labels) are out; 11 and 12 are
    # the same photo, so 12 is skipped; next largest is 13.
    assert [p for p, _ in out] == [11, 13]


def test_pinned_id_of_another_class_is_ignored(con):
    out = rg._dclass_exemplars(con, "D40", pinned=(20, 10))
    assert [p for p, _ in out] == [10, 11]


def test_n_caps_the_result(con):
    assert len(rg._dclass_exemplars(con, "D40", pinned=(), n=1)) == 1


def test_empty_class_returns_empty(con):
    assert rg._dclass_exemplars(con, "D10", pinned=(5794,)) == []


def test_box_style_math_matches_percentages():
    box = {"cx": 0.33, "cy": 0.60, "w": 0.65, "h": 0.59}
    left = 100 * (box["cx"] - box["w"] / 2)
    top = 100 * (box["cy"] - box["h"] / 2)
    assert round(left, 2) == 0.5
    assert round(top, 2) == 30.5
