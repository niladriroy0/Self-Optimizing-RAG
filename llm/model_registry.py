# llm/model_registry.py

MODEL_REGISTRY = {
    "fast": "phi3:latest",                  # ultra fast
    "cheap": "phi3:latest",
    "balanced": "mistral:latest",

    # 🔥 core intelligence layers
    "reasoning": "deepseek-r1:1.5b",
    "decomposition": "deepseek-r1:1.5b",

    # 🔥 coding
    "coding_light": "deepseek-coder:1.3b",
    "coding_heavy": "deepseek-coder:6.7b"
}


def get_model(category: str):
    return MODEL_REGISTRY.get(category, "mistral:latest")


# ----------------------------------
# MODEL CAPABILITY PROFILES
# ----------------------------------

MODEL_PROFILES = {
    "phi3:latest": {
        "cost": 1,
        "latency": 1,
        "reasoning": 2,
        "coding": 1
    },
    "mistral:latest": {
        "cost": 2,
        "latency": 2,
        "reasoning": 3,
        "coding": 2
    },
    "qwen2.5:3b": {
        "cost": 2,
        "latency": 2,
        "reasoning": 4,
        "coding": 2
    },
    "deepseek-r1:1.5b": {
        "cost": 2,
        "latency": 3,
        "reasoning": 5,
        "coding": 2
    },
    "deepseek-coder:1.3b": {
        "cost": 2,
        "latency": 2,
        "reasoning": 2,
        "coding": 4
    },
    "deepseek-coder:6.7b": {
        "cost": 4,
        "latency": 4,
        "reasoning": 3,
        "coding": 5
    }
}


# ----------------------------------
# 🔥 DYNAMIC MODEL SELECTION
# ----------------------------------

def select_best_model(task: str):

    best_model = None
    best_score = -float("inf")

    for model, profile in MODEL_PROFILES.items():

        if task == "reasoning":
            score = profile["reasoning"] - profile["latency"]

        elif task == "coding":
            score = profile["coding"] - profile["latency"]

        elif task == "fast":
            score = -profile["latency"]

        else:
            score = profile["reasoning"]

        if score > best_score:
            best_score = score
            best_model = model

    return best_model