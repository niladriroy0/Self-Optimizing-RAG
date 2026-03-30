import random
from llm.model_registry import MODEL_PROFILES, get_best_model_for_task
from control_plane.config_manager import config_manager


# ----------------------------------
# 🔥 FEATURE EXTRACTION (UPGRADED)
# ----------------------------------

def extract_features(query_analysis):

    text = query_analysis.get("text", "")

    complexity_map = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0
    }

    return {
        "length": query_analysis.get("length", 0),
        "type": query_analysis.get("type", "general"),
        "has_code": int("def" in text or "{" in text or "class" in text),

        "complexity_score": complexity_map.get(
            query_analysis.get("complexity", "medium"),
            0.5
        ),

        "ambiguity": query_analysis.get("ambiguity", 0.3),

        "requires_reasoning": int(
            query_analysis.get("type") == "reasoning"
            or query_analysis.get("is_multi_hop", False)
        ),

        "requires_coding": int(
            query_analysis.get("type") == "coding"
        ),

        # 🔥 NEW (HYBRID AWARE)
        "needs_keyword": int(query_analysis.get("needs_keyword", False)),
        "needs_semantic": int(query_analysis.get("needs_semantic", True)),
    }


# ----------------------------------
# 🔥 MODEL SCORING (IMPROVED)
# ----------------------------------

def score_model(model_name, features):

    profile = MODEL_PROFILES[model_name]

    score = 0

    # --------------------------
    # CORE TASK ALIGNMENT
    # --------------------------

    score += profile["reasoning"] * features["requires_reasoning"]
    score += profile["coding"] * (
        features["has_code"] + features["requires_coding"]
    )

    # --------------------------
    # COMPLEXITY HANDLING
    # --------------------------

    score += features["complexity_score"] * profile["reasoning"]

    # --------------------------
    # AMBIGUITY HANDLING
    # --------------------------

    score += features["ambiguity"] * profile["reasoning"]

    # --------------------------
    # HYBRID SIGNAL BOOST
    # --------------------------

    if features["needs_semantic"]:
        score += 0.5 * profile["reasoning"]

    # --------------------------
    # PENALTIES
    # --------------------------

    score -= 0.3 * profile["cost"]
    score -= 0.2 * profile["latency"]

    return score


# ----------------------------------
# 🔥 ADVANCED ROUTER
# ----------------------------------

def route_model_advanced(query_analysis):

    # 🔥 CONFIG OVERRIDE (CONTROL PLANE POWER)
    forced_model = config_manager.get_param("model")

    if forced_model in MODEL_PROFILES:
        return forced_model

    features = extract_features(query_analysis)

    best_model = None
    best_score = float("-inf")

    for model_name in MODEL_PROFILES:
        s = score_model(model_name, features)

        if s > best_score:
            best_score = s
            best_model = model_name

    return best_model


# ----------------------------------
# 🔥 EXPLORATION (SELF-OPTIMIZATION)
# ----------------------------------

def route_model_with_exploration(query_analysis, epsilon=0.1):

    models = list(MODEL_PROFILES.keys())

    if random.random() < epsilon:
        return random.choice(models)

    return route_model_advanced(query_analysis)


# ----------------------------------
# 🔥 FAST → REASONING FALLBACK DECISION
# ----------------------------------

def should_use_fast_path(query_analysis):

    return (
        query_analysis.get("complexity") == "low"
        and not query_analysis.get("is_multi_hop", False)
    )


def get_primary_model(query_analysis):

    if should_use_fast_path(query_analysis):
        return get_best_model_for_task("fast")

    return route_model_advanced(query_analysis)


def get_fallback_model(query_analysis):

    # Always use strong reasoning fallback
    return get_best_model_for_task("reasoning")