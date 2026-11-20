"""eval_real's grouping and metrics on synthetic arrays (no model, no real data)."""
import numpy as np

from floor_seg.eval_real import read_groups, scores


def test_groups_from_category_and_reason_columns(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("name,run,reason\nf1,r,disagree+narrow_0.59m\nf2,r,ordinary\nf3,r,blocked:end\n")
    b = tmp_path / "b.csv"
    b.write_text("tile,name,category\n1,f3,doorstep\n2,f4,plinth_false_floor\n")
    assert read_groups([a, b]) == {"f1": "disagree", "f2": "ordinary", "f3": "doorstep", "f4": "plinth_false_floor"}


def test_scores_pooled_and_ignore():
    y = np.array([[1, 1, 0, 255]], np.uint8)       # floor, floor, obstacle, ignore
    prob = np.array([[0.9, 0.2, 0.9, 0.9]])        # one floor found, one missed, one false floor
    iou, ff, mf, per = scores([("f", prob, y)], 0.5)
    assert (iou, ff, mf) == (1 / 3, 1.0, 0.5) and per == [(1 / 3, "f")]
    iou, ff, mf, _ = scores([("g", np.ones((1, 2)), np.ones((1, 2), np.uint8))], 0.5)
    assert iou == 1.0 and np.isnan(ff) and mf == 0.0   # no obstacle pixel: false-floor rate undefined
