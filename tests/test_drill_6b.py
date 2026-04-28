"""Autograder tests for Drill 6B — Embedding Basics."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drill import load_glove, cosine_similarity, nearest_neighbors


GLOVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "glove_5k_50d.txt")


@pytest.fixture
def embeddings():
    """Load GloVe vectors for use in tests."""
    result = load_glove(GLOVE_PATH)
    assert result is not None, "load_glove returned None"
    return result


# ── Load GloVe ───────────────────────────────────────────────────────────

def test_load_glove():
    """GloVe loader should return a dict with ~5000 words, each a 50-d array."""
    result = load_glove(GLOVE_PATH)
    assert result is not None, "load_glove returned None"
    assert isinstance(result, dict), "load_glove must return a dict"
    assert len(result) >= 4000, f"Expected ~5000 words, got {len(result)}"
    # Check a sample entry
    sample_word = next(iter(result))
    vec = result[sample_word]
    assert isinstance(vec, np.ndarray), "Values must be numpy arrays"
    assert vec.shape == (50,), f"Expected shape (50,), got {vec.shape}"


# ── Cosine Similarity ────────────────────────────────────────────────────

def test_cosine_similarity(embeddings):
    """Cosine similarity of 'king' and 'queen' should be > 0.5."""
    assert "king" in embeddings, "'king' not found in GloVe vocabulary"
    assert "queen" in embeddings, "'queen' not found in GloVe vocabulary"
    sim = cosine_similarity(embeddings["king"], embeddings["queen"])
    assert sim is not None, "cosine_similarity returned None"
    assert isinstance(sim, (float, np.floating)), "cosine_similarity must return a float"
    assert sim > 0.5, f"Expected cosine('king', 'queen') > 0.5, got {sim:.4f}"


def test_cosine_similarity_zero_norm():
    """Zero-norm input must return 0.0 (no division-by-zero blowup)."""
    zero = np.zeros(50)
    nonzero = np.ones(50)
    # Both orderings + both-zero case
    assert cosine_similarity(zero, nonzero) == 0.0, (
        "cosine_similarity(zeros, nonzero) must return 0.0 (zero-norm guard)"
    )
    assert cosine_similarity(nonzero, zero) == 0.0, (
        "cosine_similarity(nonzero, zeros) must return 0.0 (zero-norm guard)"
    )
    assert cosine_similarity(zero, zero) == 0.0, (
        "cosine_similarity(zeros, zeros) must return 0.0 (both-zero case)"
    )


def test_cosine_similarity_dissimilar(embeddings):
    """Dissimilar words should have lower similarity than similar words."""
    assert "king" in embeddings and "queen" in embeddings and "banana" in embeddings
    sim_similar = cosine_similarity(embeddings["king"], embeddings["queen"])
    sim_dissimilar = cosine_similarity(embeddings["king"], embeddings["banana"])
    assert sim_similar is not None and sim_dissimilar is not None
    assert sim_dissimilar < sim_similar, (
        f"cosine('king', 'banana')={sim_dissimilar:.4f} should be < "
        f"cosine('king', 'queen')={sim_similar:.4f}"
    )


# ── Nearest Neighbors ────────────────────────────────────────────────────

def test_nearest_neighbors(embeddings):
    """Top-5 neighbors of 'king' should include 'queen'."""
    neighbors = nearest_neighbors("king", embeddings, n=5)
    assert neighbors is not None, "nearest_neighbors returned None"
    assert isinstance(neighbors, list), "nearest_neighbors must return a list"
    assert len(neighbors) == 5, f"Expected 5 neighbors, got {len(neighbors)}"
    # Each entry should be a (word, score) tuple
    for item in neighbors:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Each neighbor must be a (word, score) tuple, got {item}"
        )
    neighbor_words = [w for w, _ in neighbors]
    assert "king" not in neighbor_words, "Query word should not appear in its own neighbors"
    # 'queen' should be among the nearest neighbors of 'king'
    assert "queen" in neighbor_words, (
        f"Expected 'queen' in top-5 neighbors of 'king', got {neighbor_words}"
    )
    # Spec: results must be sorted by similarity descending
    scores = [s for _, s in neighbors]
    assert scores == sorted(scores, reverse=True), (
        f"Neighbors must be sorted by similarity descending; got scores {scores}"
    )


def test_nearest_neighbors_n_parameter(embeddings):
    """nearest_neighbors must honor the n parameter, not hardcode 5."""
    for n in (1, 3, 10):
        neighbors = nearest_neighbors("king", embeddings, n=n)
        assert neighbors is not None, f"nearest_neighbors returned None for n={n}"
        assert len(neighbors) == n, (
            f"Expected {n} neighbors when called with n={n}, got {len(neighbors)} "
            "(function must honor the n argument, not hardcode 5)"
        )
        scores = [s for _, s in neighbors]
        assert scores == sorted(scores, reverse=True), (
            f"Neighbors must be sorted by similarity descending for n={n}"
        )
