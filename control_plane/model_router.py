# control_plane/model_router.py

import random

from llm.model_registry import MODEL_PROFILES, get_best_model_for_task
from control_plane.config_manager import config_manager


# ----------------------------------
# 🔥 FEATURE EXTRACTION
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

        "needs_keyword": int(query_analysis.get("needs_keyword", False)),
        "needs_semantic": int(query_analysis.get("needs_semantic", True)),
    }


# ----------------------------------
# 🔥 SMART SPLIT ROUTING (CORE)
# ----------------------------------

def route_by_task(query_analysis):

    q_type = query_analysis.get("type")
    complexity = query_analysis.get("complexity")

    # ----------------------------------
    # 💻 CODING ROUTING (HIGHEST PRIORITY)
    # ----------------------------------

    if q_type == "coding":

        if complexity == "low":
            return "deepseek-coder:1.3b"

        if complexity == "high":
            return "deepseek-coder:6.7b"

        # medium fallback
        return "deepseek-coder:1.3b"

    # ----------------------------------
    # ⚡ FAST PATH (SIMPLE QUERIES)
    # ----------------------------------

    if (
        complexity == "low"
        and not query_analysis.get("is_multi_hop", False)
        and q_type != "reasoning"
    ):
        return "phi3:latest"

    # ----------------------------------
    # 🧠 REASONING
    # ----------------------------------

    if q_type == "reasoning" or query_analysis.get("is_multi_hop"):
        return "qwen2.5:3b"

    return None  # fallback to scoring


# ----------------------------------
# 🔥 MODEL SCORING (FALLBACK)
# ----------------------------------

def score_model(model_name, features):

    profile = MODEL_PROFILES[model_name]

    score = 0

    # ----------------------------------
    # CODING PRIORITY
    # ----------------------------------

    if features["requires_coding"]:
        score += 2.5 * profile["coding"]
        score += 0.5 * profile["reasoning"]
    else:
        score += profile["reasoning"] * features["requires_reasoning"]

    # ----------------------------------
    # COMPLEXITY
    # ----------------------------------

    score += features["complexity_score"] * profile["reasoning"]

    # ----------------------------------
    # AMBIGUITY
    # ----------------------------------

    score += features["ambiguity"] * profile["reasoning"]

    # ----------------------------------
    # CODE SIGNAL
    # ----------------------------------

    if features["has_code"]:
        score += 1.5 * profile["coding"]

    # ----------------------------------
    # HYBRID BOOST
    # ----------------------------------

    if features["needs_semantic"]:
        score += 0.5 * profile["reasoning"]

    # ----------------------------------
    # PENALTIES
    # ----------------------------------

    score -= 0.25 * profile["cost"]
    score -= 0.15 * profile["latency"]

    return score


# ----------------------------------
# 🔥 ADVANCED ROUTER
# ----------------------------------

def route_model_advanced(query_analysis):

    # ----------------------------------
    # 🎛 CONTROL PLANE OVERRIDE
    # ----------------------------------

    forced_model = config_manager.get_param("model")

    if forced_model in MODEL_PROFILES:
        return forced_model

    # ----------------------------------
    # 🔥 TASK-BASED ROUTING FIRST
    # ----------------------------------

    task_model = route_by_task(query_analysis)

    if task_model:
        return task_model

    # ----------------------------------
    # 🔁 FALLBACK TO SCORING
    # ----------------------------------

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
# 🔥 EXPLORATION (SELF-LEARNING)
# ----------------------------------

def route_model_with_exploration(query_analysis, epsilon=0.1):

    models = list(MODEL_PROFILES.keys())

    if random.random() < epsilon:
        return random.choice(models)

    return route_model_advanced(query_analysis)


# ----------------------------------
# 🔥 PRIMARY MODEL (PIPELINE ENTRY)
# ----------------------------------

def get_primary_model(query_analysis):
    return route_model_advanced(query_analysis)


# ----------------------------------
# 🔥 FALLBACK MODEL (STRONG REASONING)
# ----------------------------------

def get_fallback_model(query_analysis):
    return get_best_model_for_task("reasoning")