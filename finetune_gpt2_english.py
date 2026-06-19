"""
A.8 — Disambiguation Crash & Recovery Analysis (H2).

Confidence is averaged in a window around the disambiguation token:
offset -1 = pre-disambiguation, 0 = at disambiguation, +1 = post-disambiguation.
"""
import numpy as np

WINDOW = 4


def recovery_analysis(records):
    """
    records: list of dicts each with keys 'offset' (int, relative to the
             disambiguation token) and 'conf_max' (float attention confidence)

    Returns: dict {offset: mean_confidence}
    """
    conf_by_offset = {o: [] for o in range(-WINDOW, WINDOW + 1)}
    for r in records:
        o = r.get("offset")
        if o is not None and not np.isnan(r["conf_max"]):
            conf_by_offset[int(o)].append(r["conf_max"])
    return {o: np.mean(v) for o, v in conf_by_offset.items() if v}


if __name__ == "__main__":
    # Pre = offset -1, At = offset 0, Post = offset +1
    pass
