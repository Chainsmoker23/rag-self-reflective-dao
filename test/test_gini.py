"""
Unit Tests for Gini Fairness
Run: pytest tests/test_gini.py -v
Verifies paper claims: e.g., equal votes Gini=0; skewed ~0.82 on sample.
"""

import pytest
import numpy as np
import pandas as pd
from pseudocode.gini_fairness import compute_gini


def test_compute_gini_equal():
    """Perfect equality: Gini=0."""
    votes = [1, 1, 1, 1]
    assert compute_gini(votes) == 0.0


def test_compute_gini_skewed():
    """Known case: [1,2,3,10] Gini=0.4375."""
    votes = [1, 2, 3, 10]
    assert np.isclose(compute_gini(votes), 0.4375, atol=1e-4)


def test_compute_gini_sample_data():
    """Sample CSV: Gini ~0.82 (exponential skew)."""
    df = pd.read_csv("../data/sample_dao_votes.csv")  # Relative path
    gini = compute_gini(df['votes'].tolist())
    assert 0.80 < gini < 0.85  # Matches baseline claim


def test_compute_gini_empty():
    """Edge: Empty list Gini=0."""
    assert compute_gini([]) == 0.0


def test_compute_gini_negative():
    """Error on negatives."""
    with pytest.raises(ValueError):
        compute_gini([-1, 2])


# Run if direct
if __name__ == "__main__":
    pytest.main(["-v"])