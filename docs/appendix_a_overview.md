# Appendix A Overview: Mapping Repo to Paper

This doc summarizes how repo files align with the paper's Appendix A (pp. 39–42: Detailed Pseudocode Blueprints).

## Key Mappings
- **compute_gini (p. 39)**: Implemented in `pseudocode/gini_fairness.py` (fixed formula; tests in `tests/test_gini.py`).
- **rag_reflect_loop (pp. 39–41)**: Full sim in `pseudocode/reflection_loop.py` (3-iter convergence with QV/RAG mocks).
- **Continuous Learning Stub (p. 42)**: Integrated into reflection loop (beta/alpha scaling for feedback).
- **Sample Data (Vignettes, p. 45)**: `data/sample_dao_votes.csv` (100 skewed votes; Gini~0.82 baseline).
- **Proof Sims (Section 5)**: `simulations/gini_convergence.ipynb` (plots Eq. 1 trajectory; verifies Theorem 2).

## Quick Validation
1. Run `python -m pseudocode.reflection_loop` → Confirms Gini <0.35 in ≤3 iters.
2. Open notebook → Visualizes +35% uplift.
3. `pytest tests/` → 100% pass rate.

Extends paper's theoretical DSR [73] with executable artifacts. For blockchain mocks (e.g., Web3 stubs), see future v1.1.

Created: October 25, 2025 | Cite: Sarkar (2025).