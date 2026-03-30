# llm/model_registry.py

# ----------------------------------
# MODEL REGISTRY
# ----------------------------------

MODEL_REGISTRY = {
    "fast": "phi3:latest",
    "cheap": "phi3:latest",
    "balanced": "mistral:latest",

    "reasoning": "qwen2.5:3b",
    "decomposition": "qwen2.5:3b",

    "coding_light": "deepseek-coder:1.3b",
    "coding_heavy": "deepseek-coder:6.7b"
}


def get_model(category: str):
    return MODEL_REGISTRY.get(category, "mistral:latest")


# ----------------------------------
# MODEL PROFILES
# ----------------------------------

MODEL_PROFILES = {
    "phi3:latest": {"cost": 1, "latency": 1, "reasoning": 2, "coding": 1},
    "mistral:latest": {"cost": 2, "latency": 2, "reasoning": 3, "coding": 2},
    "qwen2.5:3b": {"cost": 2, "latency": 2, "reasoning": 5, "coding": 2},
    "deepseek-coder:1.3b": {"cost": 2, "latency": 2, "reasoning": 2, "coding": 4},
    "deepseek-coder:6.7b": {"cost": 4, "latency": 4, "reasoning": 3, "coding": 5}
}


# ----------------------------------
# TASK → CATEGORY
# ----------------------------------

def map_task_to_category(task: str):

    mapping = {
        "fast": "fast",
        "simple": "fast",
        "balanced": "balanced",
        "general": "balanced",
        "reasoning": "reasoning",
        "complex": "reasoning",
        "decomposition": "decomposition",
        "coding": "coding_heavy"
    }

    return mapping.get(task, "balanced")


# ----------------------------------
# SMART SELECTION
# ----------------------------------

def select_best_model(task: str):

    best_model = None
    best_score = float("-inf")

    for model, p in MODEL_PROFILES.items():

        if task == "reasoning":
            score = (p["reasoning"] * 2) - p["latency"]

        elif task == "coding":
            score = (p["coding"] * 2) - p["latency"]

        elif task == "fast":
            score = -p["latency"]

        else:
            score = p["reasoning"] - p["latency"]

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


# ----------------------------------
# FINAL ENTRY
# ----------------------------------

def get_best_model_for_task(task: str):

    category = map_task_to_category(task)
    base = get_model(category)
    smart = select_best_model(task)

    return smart or base