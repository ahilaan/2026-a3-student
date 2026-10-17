"""
submission/fusion.py -- rank fusion over the named base-ranker scores that
fuse_candidates() receives (assignment Sections 3.1 and 4.1).

Input convention for every function here: `base_scores` maps each candidate
doc_id to {channel_name: score}, HIGHER = BETTER for every channel (LMIR
scores are log-probabilities, i.e. negative -- larger is still better).
Output: {doc_id: fused_score}, higher = better. Ties in a channel are broken
by the dict's insertion order (the candidate order), so results are
deterministic.

Hand-written textbook RRF, CombSUM and CombMNZ implementations.
"""
from typing import Dict, Iterable, List, Optional


def _channels(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]]) -> List[str]:
    if channels is not None:
        return list(channels)
    first = next(iter(base_scores.values()), {})
    return list(first.keys())


def rrf_scores(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]] = None,
               c: float = 60.0) -> Dict[str, float]:
    """Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
    score(d) = sum over channels of 1 / (c + rank_channel(d)), ranks 1-based,
    best score = rank 1. Needs only each channel's ORDER, never its scale."""
    fused = {d: 0.0 for d in base_scores}
    for channel in _channels(base_scores, channels):
        ranked = sorted(base_scores, key=lambda d: -base_scores[d][channel])
        for rank, doc in enumerate(ranked, start=1):
            fused[doc] += 1.0 / (c + rank)
    return fused


def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise {doc: score} to [0, 1] within one channel. A channel
    where every document has the same score normalises to all zeros."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {d: 0.0 for d in values}
    return {d: (v - lo) / (hi - lo) for d, v in values.items()}


def combsum_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombSUM (Fox & Shaw, TREC-2 1994): sum of the per-channel MIN-MAX-
    normalised scores."""
    fused = {d: 0.0 for d in base_scores}
    for channel in _channels(base_scores, channels):
        normalized = minmax_normalize({d: scores[channel] for d, scores in base_scores.items()})
        for doc, score in normalized.items():
            fused[doc] += score
    return fused


def combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """CombMNZ: CombSUM multiplied by the number of channels in which the
    document has a normalised score > 0. (Every candidate here has a score
    from every channel, so 'appears in the list' is defined as 'normalised
    score > 0' -- see docs/SUBMISSION_INTERFACE.md.)"""
    totals = {d: 0.0 for d in base_scores}
    counts = {d: 0 for d in base_scores}
    for channel in _channels(base_scores, channels):
        normalized = minmax_normalize({d: scores[channel] for d, scores in base_scores.items()})
        for doc, score in normalized.items():
            totals[doc] += score
            counts[doc] += score > 0.0
    return {d: totals[d] * counts[d] for d in base_scores}


def weighted_power_combsum(base_scores, weights, power=0.5):
    """TRAIN-tuned weighted fusion with a concave min-max normalization.

    The textbook functions above retain their original definitions. Constants
    and channel weights are selected offline using released TRAIN queries.
    """
    fused = {doc: 0.0 for doc in base_scores}
    for channel in _channels(base_scores, None):
        weight = weights.get(channel, 1.0)
        if weight == 0:
            continue
        normalized = minmax_normalize({doc: scores[channel] for doc, scores in base_scores.items()})
        for doc, value in normalized.items():
            fused[doc] += weight * value ** power
    return fused
