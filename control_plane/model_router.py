import random
from llm.model_registry import MODEL_PROFILES
from control_plane.config_manager import config_manager  # ✅ NEW


def extract_features(query_analysis):

    text = query_analysis.get("text", "")

    # 🔥 FIX: complexity normalization
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
        "requires_reasoning": int(query_analysis.get("type") == "reasoning"),
        "requires_coding": int(query_analysis.get("type") == "coding"),
    }

def score_model(model_name, features):

    profile = MODEL_PROFILES[model_name]

    score = 0

    score += profile["reasoning"] * features["requires_reasoning"]
    score += profile["coding"] * (
        features["has_code"] + features["requires_coding"]
    )
    score += features["complexity_score"] * profile["reasoning"]
    score += features["ambiguity"] * profile["reasoning"]

    score -= 0.3 * profile["cost"]
    score -= 0.2 * profile["latency"]

    return score


# ADD THIS ONLY inside route_model_advanced()

def route_model_advanced(query_analysis):

    # 🔥 FIX: only override if valid model
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


def route_model_with_exploration(query_analysis, epsilon=0.1):

    models = list(MODEL_PROFILES.keys())

    if random.random() < epsilon:
        return random.choice(models)

    return route_model_advanced(query_analysis)