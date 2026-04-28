"""
RRF (Reciprocal Rank Fusion) implementations for RAG systems.

This module provides various strategies for fusing ranked retrieval results.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

RRFStrategy = Literal["best_rank", "diminishing_returns", "max_plus_bonus", "soft_dedup", "standard"]


@dataclass
class FusionConfig:
    """Configuration for RRF fusion."""

    strategy: RRFStrategy = "standard"
    k: int = 60
    lambda_param: float = 0.3
    alpha: float = 0.5
    beta: float = 0.1
    decay: Literal["harmonic", "exponential"] = "exponential"


def best_rank_aggregation(
    document_rankings: dict[str, list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Strategy 1: Best-rank aggregation.

    Takes the best rank for each document across all retrievers.
    score(d) = 1 / (k + min_rank)

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs (ordered by rank)
        k: RRF constant (higher = more weight on lower ranks)

    Returns:
        Dict mapping document ID to fused score
    """
    doc_best_ranks: dict[str, dict[str, int]] = {}

    for retriever, doc_ids in document_rankings.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id not in doc_best_ranks:
                doc_best_ranks[doc_id] = {}
            if retriever not in doc_best_ranks[doc_id]:
                doc_best_ranks[doc_id][retriever] = rank
            else:
                doc_best_ranks[doc_id][retriever] = min(doc_best_ranks[doc_id][retriever], rank)

    scores = {}
    for doc_id, retriever_ranks in doc_best_ranks.items():
        scores[doc_id] = float(np.sum(1.0 / (k + np.array(list(retriever_ranks.values())))))
    return scores


def diminishing_returns(
    document_rankings: dict[str, list[str]],
    k: int = 60,
    decay: Literal["harmonic", "exponential"] = "exponential",
    alpha: float = 0.5,
    lambda_param: float = 0.3,
) -> dict[str, float]:
    """
    Strategy 2: Diminishing returns with decay weights.

    score(d) = sum_j (1 / (k + r_j) * w_j)

    Where w_j is the decay weight for the j-th rank of document d.

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs
        k: RRF constant
        decay: "harmonic" or "exponential"
        alpha: Decay factor for exponential decay (default 0.5)
        lambda_param: Weight scaling factor

    Returns:
        Dict mapping document ID to fused score
    """
    doc_ranks: dict[str, dict[str, list[int]]] = {}

    for retriever, doc_ids in document_rankings.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
            if retriever not in doc_ranks[doc_id]:
                doc_ranks[doc_id][retriever] = []
            doc_ranks[doc_id][retriever].append(rank)

    scores = {}
    for doc_id, retriever_ranks in doc_ranks.items():
        ranks_array = np.sort(np.concatenate(list(retriever_ranks.values())))
        n = len(ranks_array)

        if decay == "harmonic":
            weights = 1.0 / np.arange(1, n + 1)
        else:
            weights = alpha ** np.arange(n)

        raw_scores = 1.0 / (k + ranks_array)
        scores[doc_id] = float(np.sum(raw_scores * weights) * lambda_param)

    return scores


def max_plus_bonus(
    document_rankings: dict[str, list[str]],
    k: int = 60,
    lambda_param: float = 0.3,
) -> dict[str, float]:
    """
    Strategy 3: Max + bonus formulation.

    score(d) = 1 / (k + r_min) + λ * sum_{j>1} (1 / (k + r_j))

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs
        k: RRF constant
        lambda_param: Bonus weight for additional ranks (default 0.2-0.4)

    Returns:
        Dict mapping document ID to fused score
    """
    doc_ranks: dict[str, dict[str, list[int]]] = {}

    for retriever, doc_ids in document_rankings.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
            if retriever not in doc_ranks[doc_id]:
                doc_ranks[doc_id][retriever] = []
            doc_ranks[doc_id][retriever].append(rank)

    scores = {}
    for doc_id, retriever_ranks in doc_ranks.items():
        ranks_array = np.sort(np.concatenate(list(retriever_ranks.values())))
        base_score = 1.0 / (k + ranks_array[0])
        bonus_score = np.sum(1.0 / (k + ranks_array[1:])) if len(ranks_array) > 1 else 0.0
        scores[doc_id] = float(base_score + lambda_param * bonus_score)

    return scores


def soft_dedup_rank_inflation(
    document_rankings: dict[str, list[str]],
    k: int = 60,
    beta: float = 0.1,
) -> dict[str, float]:
    """
    Strategy 4: Soft dedup via rank inflation.

    Penalize repeated documents by inflating their ranks:
    r' = r * (1 + β * occurrence_index)

    Then apply standard RRF.

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs
        k: RRF constant
        beta: Inflation factor (higher = more penalty for duplicates)

    Returns:
        Dict mapping document ID to fused score
    """
    inflated_scores: dict[str, float] = {}

    for _retriever, doc_ids in document_rankings.items():
        if not doc_ids:
            continue

        ranks = np.arange(1, len(doc_ids) + 1, dtype=float)

        # Vectorized: find positions and counts for each unique doc
        unique_docs, first_inds, counts = np.unique(doc_ids, return_index=True, return_counts=True)

        # Compute inflation for all positions via vectorization
        all_inflated = np.zeros(len(doc_ids), dtype=float)
        for doc, first_ind, count in zip(unique_docs, first_inds, counts):
            positions = np.arange(first_ind, first_ind + count)
            inflated = ranks[positions] * (1 + beta * np.arange(count, dtype=float))
            all_inflated[positions] = inflated

        # Accumulate scores using numpy addition per unique doc
        for doc, first_ind, count in zip(unique_docs, first_inds, counts):
            positions = np.arange(first_ind, first_ind + count)
            doc_score = float(np.sum(1.0 / (k + all_inflated[positions])))
            inflated_scores[str(doc)] = inflated_scores.get(str(doc), 0.0) + doc_score

    return inflated_scores


def standard_rrf(
    document_rankings: dict[str, list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Standard RRF at document level.

    score(d) = sum_r (1 / (k + rank_r(d)))

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs (ordered by rank)
        k: RRF constant (typically 60)

    Returns:
        Dict mapping document ID to fused score
    """
    doc_ranks: dict[str, dict[str, int]] = {}

    for retriever, doc_ids in document_rankings.items():
        for rank, doc_id in enumerate(doc_ids, start=1):
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {}
            doc_ranks[doc_id][retriever] = rank

    scores = {}
    for doc_id, retriever_ranks in doc_ranks.items():
        scores[doc_id] = float(np.sum(1.0 / (k + np.array(list(retriever_ranks.values())))))

    return scores


def fuse_retriever_results(
    document_rankings: dict[str, list[str]],
    config: FusionConfig | None = None,
) -> dict[str, float]:
    """
    Fuse results from multiple retrievers using the specified strategy.

    Args:
        document_rankings: Dict mapping retriever name to list of doc IDs
        config: Fusion configuration (uses defaults if None)

    Returns:
        Dict of doc_id -> score, ordered by score descending
    """
    if config is None:
        config = FusionConfig()

    if config.strategy == "best_rank":
        scores = best_rank_aggregation(
            document_rankings,
            k=config.k,
        )
    elif config.strategy == "diminishing_returns":
        scores = diminishing_returns(
            document_rankings,
            k=config.k,
            alpha=config.alpha,
            lambda_param=config.lambda_param,
        )
    elif config.strategy == "max_plus_bonus":
        scores = max_plus_bonus(
            document_rankings,
            k=config.k,
            lambda_param=config.lambda_param,
        )
    elif config.strategy == "soft_dedup":
        scores = soft_dedup_rank_inflation(
            document_rankings,
            k=config.k,
            beta=config.beta,
        )
    elif config.strategy == "standard":
        scores = standard_rrf(document_rankings, k=config.k)
    else:
        raise ValueError(
            f"Unknown strategy: {config.strategy}. "
            f"Choose from: best_rank, diminishing_returns, max_plus_bonus, soft_dedup, standard"
        )

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
