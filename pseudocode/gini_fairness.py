
"""
Gini Fairness Computation Module
From paper Appendix A.1: Computes Gini coefficient for DAO voting fairness.
Fixed implementation: Standard formula without extra /n division.
Target: Gini <0.35 post-QV + reflection (Theorem 2).
Usage: compute_gini([1,2,3,10]) -> 0.4375
"""

import numpy as np
import pandas as pd


def compute_gini(votes):
    """
    Standard Gini coefficient formula.
    G = sum((2*i - n - 1) * x_(i)) / (n * sum(x)) where x sorted ascending.
    :param votes: List or array of vote values (non-negative).
    :return: Gini (0=perfect equality, 1=max inequality).
    """
    votes = np.array(votes)
    if len(votes) == 0:
        return 0.0
    if np.any(votes < 0):
        raise ValueError("Votes must be non-negative.")
    
    n = len(votes)
    sorted_votes = np.sort(votes)
    index = np.arange(1, n + 1)
    numerator = np.sum((2 * index - n - 1) * sorted_votes)
    denominator = n * np.sum(sorted_votes)
    return numerator / denominator


def apply_quadratic_voting(votes, credits_per_voter=10):
    """
    Simplified Quadratic Voting (QV) stub to reduce plutocracy.
    Each voter allocates 'credits' quadratically (cost = votes^2).
    From paper [91]: Mitigates whale dominance, targeting Gini reduction by ~50%.
    :param votes: Initial linear votes per voter.
    :param credits_per_voter: Fixed credits (e.g., 10 signals).
    :return: Adjusted quadratic votes list.
    """
    votes = np.array(votes)
    n = len(votes)
    adjusted = np.zeros(n)
    for i in range(n):
        # Solve v^2 <= credits for max v (integer)
        max_v = int(np.sqrt(credits_per_voter))
        adjusted[i] = min(votes[i], max_v) ** 2  # Quadratic signal strength
    return adjusted.tolist()


# Example/CLI entrypoint
if __name__ == "__main__":
    # Test with sample
    test_votes = [1, 2, 3, 10]
    gini = compute_gini(test_votes)
    print(f"Test Gini ({test_votes}): {gini:.4f}")  # 0.4375
    
    # Load sample CSV
    try:
        df = pd.read_csv("../../data/sample_dao_votes.csv")  # Relative path
        sample_gini = compute_gini(df['votes'].tolist())
        print(f"Sample DAO Gini: {sample_gini:.3f}")  # ~0.820
    except FileNotFoundError:
        print("Run from repo root or provide data path.")