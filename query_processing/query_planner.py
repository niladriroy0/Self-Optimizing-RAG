from typing import List


def decompose_query(query: str) -> List[str]:
    """
    Basic multi-hop decomposition

    Handles:
    - comparisons
    - multi-part questions
    - chained queries
    """

    q = query.lower()

    # ----------------------------------
    # 1. Comparison queries
    # ----------------------------------
    if "compare" in q or "difference between" in q:
        parts = query.replace("compare", "").split("and")
        return [p.strip() for p in parts if p.strip()]

    # ----------------------------------
    # 2. Multi-part queries
    # ----------------------------------
    if " and " in q:
        parts = query.split(" and ")
        return [p.strip() for p in parts if p.strip()]

    # ----------------------------------
    # 3. Sequential reasoning
    # ----------------------------------
    if " then " in q:
        parts = query.split(" then ")
        return [p.strip() for p in parts if p.strip()]

    # ----------------------------------
    # Default → single query
    # ----------------------------------
    return [query]