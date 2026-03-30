# query_processing/query_analyzer.py

def analyze_query(query: str):

    q = query.lower()
    words = q.split()

    # ----------------------------------
    # BASE STRUCTURE
    # ----------------------------------

    analysis = {
        "text": query,
        "length": len(words),
        "type": "general",
        "complexity": "medium",
        "is_multi_hop": False,

        # 🔥 NEW (HYBRID SIGNALS)
        "needs_keyword": False,
        "needs_semantic": True,

        # 🔥 NEW (ROUTING SIGNALS)
        "ambiguity": 0.3,
    }

    # ----------------------------------
    # TYPE DETECTION
    # ----------------------------------

    if any(w in q for w in ["code", "python", "bug", "algorithm"]):
        analysis["type"] = "coding"

    elif any(w in q for w in ["explain", "why", "how", "reason"]):
        analysis["type"] = "reasoning"

    # ----------------------------------
    # COMPLEXITY DETECTION
    # ----------------------------------

    if len(words) < 5:
        analysis["complexity"] = "low"

    elif len(words) > 15 or any(w in q for w in ["compare", "difference", "architecture"]):
        analysis["complexity"] = "high"

    # ----------------------------------
    # MULTI-HOP DETECTION
    # ----------------------------------

    if any(k in q for k in [" and ", " then ", "compare", "difference", "vs"]):
        analysis["is_multi_hop"] = True

    # ----------------------------------
    # HYBRID SIGNALS
    # ----------------------------------

    if any(w in q for w in ["id", "exact", "syntax", "error code"]):
        analysis["needs_keyword"] = True

    if any(w in q for w in ["explain", "why", "how"]):
        analysis["needs_semantic"] = True

    # ----------------------------------
    # AMBIGUITY ESTIMATION
    # ----------------------------------

    if len(words) < 4:
        analysis["ambiguity"] = 0.6

    elif "it" in q or "this" in q:
        analysis["ambiguity"] = 0.8

    return analysis