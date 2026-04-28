"""
Demo of RRF (Reciprocal Rank Fusion) strategies for RAG systems.

This demonstrates various fusion strategies for combining ranked results
from multiple retrievers at the document level.
"""

from rrf import FusionConfig, RRFStrategy, fuse_retriever_results

# Example: 5 retrieval channels ranking 10 documents
DOCUMENT_RANKINGS: dict[str, list[str]] = {
    "bm25": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    "dense": ["b", "a", "e", "c", "d", "f", "g", "j", "i", "h"],
    "sparse": ["c", "a", "b", "f", "d", "e", "h", "g", "j", "i"],
    "knowledge": ["d", "e", "a", "b", "c", "g", "f", "h", "i", "j"],
    "colaborative": ["i", "j", "h", "g", "f", "e", "d", "c", "b", "a"],
}


def _top(fused: dict[str, float], k: int = 5) -> str:
    """Get top k doc_id(score) pairs from fused results."""
    items = list(fused.items())[:k]
    return " ".join([f"{d}({s:.3f})" for d, s in items])


def demo_strategy_comparison() -> None:
    """Compare all strategies on the example rankings."""
    print("=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    strategies: list[RRFStrategy] = [
        "best_rank",
        "diminishing_returns",
        "max_plus_bonus",
        "soft_dedup",
        "standard",
    ]

    print(f"{'Strategy':<35} {'#1':<12} {'#2':<12} {'#3':<12}")
    print("-" * 72)

    for strategy in strategies:
        config = FusionConfig(strategy=strategy)
        fused = fuse_retriever_results(DOCUMENT_RANKINGS, config)
        print(f"{strategy:<35} {_top(fused)}")


def demo_parameter_tuning() -> None:
    """Demonstrate how parameters affect each strategy."""
    print()
    print("=" * 60)
    print("PARAMETER TUNING")
    print("=" * 60)

    configs: list[tuple[str, FusionConfig]] = [
        ("Diminishing Returns (exp, α=0.3)", FusionConfig(strategy="diminishing_returns", alpha=0.3)),
        ("Diminishing Returns (exp, α=0.7)", FusionConfig(strategy="diminishing_returns", alpha=0.7)),
        ("Diminishing Returns (harmonic)", FusionConfig(strategy="diminishing_returns", decay="harmonic")),
        ("Max + Bonus (λ=0.2)", FusionConfig(strategy="max_plus_bonus", lambda_param=0.2)),
        ("Max + Bonus (λ=0.4)", FusionConfig(strategy="max_plus_bonus", lambda_param=0.4)),
        ("Soft Dedup (β=0.1)", FusionConfig(strategy="soft_dedup", beta=0.1)),
        ("Soft Dedup (β=0.2)", FusionConfig(strategy="soft_dedup", beta=0.2)),
    ]

    print(f"{'Strategy':<35} {'#1':<12} {'#2':<12} {'#3':<12}")
    print("-" * 72)

    for name, config in configs:
        fused = fuse_retriever_results(DOCUMENT_RANKINGS, config)
        print(f"{name:<35} {_top(fused)}")


if __name__ == "__main__":
    print("RRF (Reciprocal Rank Fusion) Demo for RAG Systems")
    print("=" * 60)
    print(f"5 retrieval channels: {list(DOCUMENT_RANKINGS.keys())}")

    demo_strategy_comparison()
    demo_parameter_tuning()