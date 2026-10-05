#!/usr/bin/env python3
"""Record linkage for board rows and receipts: blocking then Fellegi-Sunter matching.

The method is two-stage, which is the standard record-linkage architecture
(Christen, *Data Matching*, Springer 2012, ch. 4: blocking reduces the O(n^2)
pair space to candidate pairs, matching scores each candidate).

Blocking
--------
`simhash` is Charikar's fingerprint: split a document into weighted feature
shingles, hash each, and keep the sign of each of 64 hash bits across the
shingles. Two documents with equal feature *sets* have equal fingerprints. Near
duplicates that reorder or drop a few features land within a small Hamming
radius. `lsh_bands` is Manku-Jain-das Sarma banding: split the 64-bit word into
`b` bands of `r` bits and hash each band; two fingerprints collide in a band iff
their Hamming distance is below `b - (b-1)/r` in expectation, so band-key
equality is the cheap candidate generator and Hamming distance is the expensive
scorer.

Matching
--------
Fellegi-Sunter (1969) scores each surviving candidate by agreement on a set of
comparisons c: weight(c) = log2(m_c / u_c), and the pair score is the sum. u_c is
estimated from random pairs (the U-cube of Cohen-Ravikumar-Fienberg 2003).
m_c is estimated on a *declared seed set* and is therefore only as good as that
seed; `m_seed_kind` records what the seed was. When no seed is supplied every m_c
is set to the u_c of its own comparison, which makes every weight exactly zero --
an index, not a linkage, and the caller is told so by `degenerate=True`.

The R[x]/(x^2 - sigma) relation to this file
-------------------------------------------
The 64-bit fingerprint is a projection of a sparse feature vector onto
{+1,-1}^64 read through an inner product; similarity is that inner product
normalised. It is NOT a ring cell, has no norm form, no idempotent, and no
elliptic/parabolic/hyperbolic face: the trichotomy classifies the sign of sigma
in R[x]/(x^2 - sigma), and nothing here computes a sign of sigma or consumes a
ring structure. `coldkv@9059eba bench/coldcache.py:332` uses the same object --
a fixed Rademacher projection sign(R(k - mu)) -- as an LSH recall code over
dense embeddings, which confirms the characterisation and is a different input
regime (continuous vectors) from shingled text. The primitive (a 64-bit hash of
a token) *is* subsumable by the apeiron `Cell<i64>` word hash; the index built
on top consumes only bits, so no refinement theorem is available or claimed and
no grade above `measured` attaches to any number produced here.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata

__all__ = [
    "normalize",
    "shingles",
    "simhash",
    "hamming",
    "lsh_bands",
    "blocking_pairs",
    "jaccard",
    "fellegi_sunter",
    "link_rows",
]

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    NFKD + combining-mark removal so `ReadLoop.lean` and `ReadLoop lean` do not
    differ by markup alone; `_` is kept because it separates identifiers.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(_WORD.findall(stripped.lower()))


def shingles(text: str, k: int = 3) -> dict[str, int]:
    """Word k-shingles with term frequency. Empty input yields the empty map.

    k=3 is the value Broder's shingling analysis and every downstream MinHash
    recipe use for prose; rows here are short, so the 3-shingle of a title keeps
    more signal than unigrams.
    """
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words): 1} if words else {}
    out: dict[str, int] = {}
    for i in range(len(words) - k + 1):
        out[" ".join(words[i : i + k])] = out.get(" ".join(words[i : i + k]), 0) + 1
    return out


def _token_hash(token: str) -> int:
    """64-bit token hash. sha256 prefix: deterministic, stdlib, no PYTHONHASHSEED."""
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")


def simhash(text: str, bits: int = 64) -> int:
    """Charikar's SimHash over 3-shingles.

    Each of `bits` hash positions keeps the sign of the summed tf-weight of the
    shingles whose digest has that bit set; a tie keeps the bit clear. Empty
    input returns 0, which shares a fingerprint with every other empty input --
    the caller must not treat that as a match.
    """
    feats = shingles(text)
    if not feats:
        return 0
    acc = [0] * bits
    for shingle, weight in feats.items():
        digest = _token_hash(shingle)
        for pos in range(bits):
            if digest >> pos & 1:
                acc[pos] += weight
            else:
                acc[pos] -= weight
    out = 0
    for pos in range(bits):
        if acc[pos] > 0:
            out |= 1 << pos
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def lsh_bands(fp: int, bands: int = 8, rows: int = 8) -> list[int]:
    """Manku-Jain-das Sarma band keys: `rows`-bit chunks of the 64-bit word.

    Expected detection threshold: two fingerprints differing in d of 64 bits
    share a band iff some band avoids all d differing positions, at probability
    1 - (1 - (1/2)^rows)^bands per band; the sharpest estimate of the standard
    guarantee d <= b - (b-1)/r needs d <= 8 for bands=rows=8, i.e. Jaccard
    above about 0.75 on equal feature sets. Rows shorter or less similar are
    reachable only by the exact and token-Jaccard passes, which is stated in the
    report rather than hidden.
    """
    if rows <= 0 or bands <= 0 or rows * bands > 64:
        raise ValueError(f"bands*rows must fit 64 bits: {bands}x{rows}")
    out = []
    mask = (1 << rows) - 1
    for b in range(bands):
        chunk = (fp >> (b * rows)) & mask
        out.append(int.from_bytes(hashlib.sha256(f"band{b}:{chunk}".encode()).digest()[:8], "big"))
    return out


def jaccard(a: dict[str, int], b: dict[str, int]) -> float:
    """Set Jaccard over feature keys (2|intersection| / |union|)."""
    ka, kb = set(a), set(b)
    if not ka and not kb:
        return 1.0
    inter = len(ka & kb)
    if not (ka | kb):
        return 0.0
    return 2 * inter / (len(ka) + len(kb))


def blocking_pairs(records: dict[str, str], bands: int = 8, rows: int = 8) -> set[tuple[str, str]]:
    """Candidate pairs from LSH banding, plus exact normalized-name collisions.

    A record whose SimHash is 0 (empty text) is not indexed: it would collide
    with every other empty record and manufacture candidates.
    """
    buckets: dict[tuple[int, int], list[str]] = {}
    for rid, text in records.items():
        fp = simhash(text)
        if fp == 0:
            continue
        for bi, key in enumerate(lsh_bands(fp, bands, rows)):
            buckets.setdefault((bi, key), []).append(rid)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) > 400:  # a bucket this wide is not a candidate set, it is a scan
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                pairs.add((a, b))
    by_name: dict[str, list[str]] = {}
    for rid, text in records.items():
        by_name.setdefault(normalize(text), []).append(rid)
    for members in by_name.values():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    pairs.add((a, b))
    return pairs


# --- Fellegi-Sunter matching ------------------------------------------------

COMPARISONS = ("exact_name", "prefix", "jaccard", "desc_jaccard")

# Agreement thresholds. Named constants, not tuned: 0.80 on 3-shingle Jaccard is
# the conventional near-duplicate cut and 0.60 on 2-shingles over descriptions
# is where two descriptions of the same row stop overlapping by chance.
JACCARD_NAME = 0.80
JACCARD_DESC = 0.60

# Only `exact_name` is high precision. A prefix over the row's *identifier head*
# is not: every apeiron row normalizes to a leading `apeiron` then its epic code,
# so a two-token prefix agrees for every pair in the project and carries no
# information -- measured at 377 of 7,260 pairs reported on the live board before
# this was fixed. `descriptive_head` strips the identifier head first.
#
# A pair needs a high-precision comparison or at least MIN_IDENTIFIED_AGREEMENTS
# agreements: one weak agreement is a shared template, not a link.
STRONG = ("exact_name",)
PREFIX_TOKENS = 2
MIN_IDENTIFIED_AGREEMENTS = 2

_ID_HEAD = re.compile(r"^(?:[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\s+)+")


def descriptive_head(text: str) -> str:
    """Normalized name with its leading ALL-CAPS identifier head removed.

    `APEIRON-B-B2 SIMD ctrl/tag glance` -> `simd ctrl tag glance`. The head is
    the row's identity, so comparing it says only that two rows belong to the
    same project and wave.
    """
    head = _ID_HEAD.match(text.strip())
    return normalize(text[head.end():] if head else text)

# The decision rule is Fellegi-Sunter's own (1969, "Decision rules"): a pair is a
# match when the posterior P(match | comparisons) exceeds 0.5, which is the
# natural "more likely a match than not" cut and is not a tuned threshold. The
# prior is a genuine input the method does not supply -- Fellegi-Sunter require
# the user to declare it -- so PRIOR_DUPLICATE_RATE is declared here and is a
# design assumption, grade `model`: a hand-seeded board plausibly carries one
# duplicate row per ten.
#
# The reported quantity is prior-free as well: `break_even_prior` is the prior at
# which the pair flips decision, so a pair is reported with the sensitivity it
# was reported at and a reader can re-threshold it without re-running.
PRIOR_DUPLICATE_RATE = 0.10
POSTERIOR_REPORT = 0.5


def comparison_vector(text_a: str, text_b: str, desc_a: str = "", desc_b: str = "") -> list[str]:
    """The comparisons the matcher scores; see module docstring for the method."""
    na, nb = normalize(text_a), normalize(text_b)
    out: list[str] = []
    if na == nb:
        out.append("exact_name")
    ta = descriptive_head(text_a).split()
    tb = descriptive_head(text_b).split()
    if len(ta) >= PREFIX_TOKENS and len(tb) >= PREFIX_TOKENS and ta[:PREFIX_TOKENS] == tb[:PREFIX_TOKENS]:
        out.append("prefix")
    if jaccard(shingles(text_a), shingles(text_b)) >= JACCARD_NAME:
        out.append("jaccard")
    da, db = shingles(desc_a, k=2), shingles(desc_b, k=2)
    if da and db and jaccard(da, db) >= JACCARD_DESC:
        out.append("desc_jaccard")
    return out


def fellegi_sunter(
    comparisons: list[str],
    m: dict[str, float],
    u: dict[str, float],
    identified: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    """Sum of log2(m_c/u_c) over the *identified* comparisons.

    A comparison whose u estimate is exactly zero was not observed once in
    `sample` random pairs; at n = 118 rows the expected count for exact-name
    agreement is zero, so its u is not estimable. Clamping u to 1e-4 -- the usual
    fix -- produces a weight of log2(1/1e-4) = 13.3 bits that is an artifact of
    the clamp and not of the data. Such a comparison is therefore *excluded*
    from the score and listed as `not_identified` instead. The sum is over
    identified comparisons only, so every bit in it is data-derived.
    """
    per: dict[str, float] = {}
    total = 0.0
    for c in COMPARISONS:
        if c not in identified:
            per[c] = None
            continue
        p = min(max(m[c] if c in comparisons else u[c], 1e-4), 1 - 1e-4)
        uu = min(max(u[c], 1e-4), 1 - 1e-4)
        weight = math.log2(p / uu)
        per[c] = round(weight, 3)
        total += weight
    return total, per


def estimate_u(
    names: dict[str, str],
    descs: dict[str, str],
    sample: int = 4000,
    seed: int = 20261004,
) -> tuple[dict[str, float], tuple[str, ...]]:
    """U-cube: per-comparison agreement rate over uniformly random pairs.

    Cohen-Ravikumar-Fienberg (KDD workshop 2003) estimate the non-match
    distribution by sampling pairs that are assumed non-matching; uniform random
    pairs are that estimator when matches are rare relative to the population,
    which is the stated assumption here and is the assumption's weakness.

    Returns (u, identified): a comparison is identified when it agreed at least
    once in `sample` pairs, since a zero count carries no information about the
    rate.
    """
    rng = random.Random(seed)
    ids = sorted(names)
    if len(ids) < 2:
        return {c: 0.5 for c in COMPARISONS}, tuple(COMPARISONS)
    counts = {c: 0 for c in COMPARISONS}
    for _ in range(sample):
        a, b = rng.sample(ids, 2)
        for c in comparison_vector(names[a], names[b], descs.get(a, ""), descs.get(b, "")):
            counts[c] += 1
    u = {c: counts[c] / sample for c in COMPARISONS}
    identified = tuple(c for c in COMPARISONS if counts[c] > 0)
    return u, identified


def estimate_m(
    seed_pairs: list[tuple[str, str, str, str]],
    u: dict[str, float],
) -> tuple[dict[str, float], str]:
    """Match distribution from a declared seed set; returns (m, seed description).

    The seed is (name_a, name_b, desc_a, desc_b): descriptions travel with it,
    because scoring the seed with empty descriptions would leave the description
    comparison unestimated and silently zero its weight.

    A seed pair here is an exact normalized-name collision, which is a match by
    construction of the test rather than by judgement. If the seed is empty every
    m_c equals its u_c, every weight is 0, and the reported set is the
    high-precision comparisons alone -- an index, not a linkage, and the seed
    description says so in words.
    """
    if not seed_pairs:
        return dict(u), "none (degenerate: all weights zero)"
    counts = {c: 0 for c in COMPARISONS}
    for a, b, da, db in seed_pairs:
        for c in comparison_vector(a, b, da, db):
            counts[c] += 1
    m = {c: max(counts[c] / len(seed_pairs), 1e-3) for c in COMPARISONS}
    for c in COMPARISONS:
        if counts[c] == 0:
            m[c] = u[c]
    return m, f"exact-name seed, n={len(seed_pairs)}"


def posterior(
    comparisons: list[str],
    m: dict[str, float],
    u: dict[str, float],
    identified: tuple[str, ...],
    prior: float = PRIOR_DUPLICATE_RATE,
) -> tuple[float, float, float]:
    """P(match | comparisons) over the identified comparisons.

    Returns (posterior, score, break_even_prior) where the score is the
    Fellegi-Sunter log-likelihood ratio in bits and break_even_prior is the
    prior at which the posterior equals POSTERIOR_REPORT. A comparison with an
    unestimated u is skipped in both the match and the non-match likelihood, so
    it cannot move the ratio; its probability is the same on both hypotheses.
    """
    log_m, log_u, lm, lu = 0.0, 0.0, 1.0, 1.0
    for c in COMPARISONS:
        if c not in identified:
            continue
        uu = min(max(u[c], 1e-9), 1 - 1e-9)
        lu *= uu
        if c in comparisons:
            pp = min(max(m[c], 1e-9), 1 - 1e-9)
        else:
            pp = uu
        lm *= pp
        log_m += math.log2(pp)
        log_u += math.log2(uu)
    prior = min(max(prior, 1e-9), 1 - 1e-9)
    num = prior * lm
    den = num + (1 - prior) * lu
    post = num / den if den else 0.0
    flip = lu / (lm + lu) if (lm + lu) else 1.0
    return post, (log_m - log_u), flip


def link_rows(
    names: dict[str, str],
    descs: dict[str, str],
    bands: int = 8,
    rows: int = 8,
    exhaustive_limit: int = 1500,
) -> dict:
    """Blocking then Fellegi-Sunter matching. Read-only; returns a report dict.

    `exhaustive_limit` selects the mode and is reported. At n rows the pair
    space is n(n-1)/2 -- 6,903 at the 118 rows this corpus carries, which a
    Python loop scores in well under a second, so below the limit every pair is
    scored and blocking is *not* used: a blocking scheme that can miss a pair
    is a liability with nothing to buy. Above the limit the LSH bands supply
    the candidates and the recall floor is the banding guarantee in
    `lsh_bands`, not a scan.
    """
    text = {rid: f"{names[rid]} {normalize(descs.get(rid, ''))[:200]}" for rid in names}
    ids = sorted(names)
    exhaustive = len(ids) <= exhaustive_limit
    if exhaustive:
        candidates = {(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]}
        mode = f"exhaustive ({len(candidates)} pairs, n={len(ids)} <= {exhaustive_limit})"
    else:
        candidates = blocking_pairs(text, bands, rows)
        mode = f"lsh banding ({bands}x{rows} bits, recall floor stated in lsh_bands)"
    by_norm: dict[str, list[str]] = {}
    for rid, nm in names.items():
        by_norm.setdefault(normalize(nm), []).append(rid)
    seed_pairs = [
        (a, b)
        for members in by_norm.values()
        if len(members) > 1
        for i, a in enumerate(members)
        for b in members[i + 1 :]
    ]
    u, identified = estimate_u(names, descs)
    m, seed_kind = estimate_m(
        [(names[a], names[b], descs.get(a, ""), descs.get(b, "")) for a, b in seed_pairs], u
    )
    scored = []
    for a, b in sorted(candidates):
        comps = comparison_vector(names[a], names[b], descs.get(a, ""), descs.get(b, ""))
        post, score, flip = posterior(comps, m, u, identified)
        _, per = fellegi_sunter(comps, m, u, identified)
        scored.append(
            {
                "a": a,
                "b": b,
                "score": round(score, 3),
                "posterior": round(post, 4),
                "break_even_prior": round(flip, 4),
                "comparisons": comps,
                "weights": per,
                "hamming": hamming(simhash(names[a]), simhash(names[b])),
            }
        )
    reported = [
        s for s in scored
        if (
            any(c in s["comparisons"] for c in STRONG)
            or (
                s["posterior"] >= POSTERIOR_REPORT
                and len([c for c in s["comparisons"] if c in identified])
                >= MIN_IDENTIFIED_AGREEMENTS
            )
        )
    ]
    reported.sort(key=lambda r: (r["break_even_prior"], -r["score"], r["a"], r["b"]))
    return {
        "mode": mode,
        "candidates": len(candidates),
        "reported": len(reported),
        "pairs": reported,
        "u": {c: round(v, 5) for c, v in u.items()},
        "m": {c: round(v, 4) for c, v in m.items()},
        "identified": list(identified),
        "not_identified": [c for c in COMPARISONS if c not in identified],
        "m_seed_kind": seed_kind,
        "prior_duplicate_rate": PRIOR_DUPLICATE_RATE,
        "posterior_report": POSTERIOR_REPORT,
        "decision_rule": "posterior >= 0.5 (Fellegi-Sunter decision rule), or a high-precision comparison agrees",
    }