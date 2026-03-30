MODEL_REGISTRY = {
    # ⚡ SPEED LAYER
    "fast": "phi3:latest",
    "cheap": "phi3:latest",

    # ⚖️ BALANCED
    "balanced": "mistral:latest",

    # 🧠 INTELLIGENCE LAYERS
    "reasoning": "qwen2.5:3b",
    "decomposition": "qwen2.5:3b",

    # 💻 CODING
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
        "reasoning": 5,   # 🔥 BEST reasoning in your stack
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
# 🔥 TASK → MODEL CATEGORY MAPPING
# ----------------------------------

def map_task_to_category(task: str):

    if task in ["simple", "fast"]:
        return "fast"

    if task in ["general", "balanced"]:
        return "balanced"

    if task in ["reasoning", "complex"]:
        return "reasoning"

    if task == "decomposition":
        return "decomposition"

    if task == "coding":
        return "coding_heavy"

    return "balanced"


# ----------------------------------
# 🔥 SMART MODEL SELECTION (PROFILE-BASED)
# ----------------------------------

def select_best_model(task: str):

    best_model = None
    best_score = -float("inf")

    for model, profile in MODEL_PROFILES.items():

        # --------------------------
        # TASK-BASED SCORING
        # --------------------------

        if task == "reasoning":
            score = (profile["reasoning"] * 2) - profile["latency"]

        elif task == "coding":
            score = (profile["coding"] * 2) - profile["latency"]

        elif task == "fast":
            score = -profile["latency"]

        elif task == "balanced":
            score = profile["reasoning"] - (0.5 * profile["latency"])

        elif task == "decomposition":
            score = (profile["reasoning"] * 2) - profile["latency"]

        else:
            score = profile["reasoning"] - profile["latency"]

        # --------------------------
        # BEST MODEL UPDATE
        # --------------------------

        if score > best_score:
            best_score = score
            best_model = model

    return best_model


# ----------------------------------
# 🔥 FINAL ROUTER (USED BY SYSTEM)
# ----------------------------------

def get_best_model_for_task(task: str):

    # Step 1: Map task → category
    category = map_task_to_category(task)

    # Step 2: Get default model
    base_model = get_model(category)

    # Step 3: Override with smarter selection
    smart_model = select_best_model(task)

    return smart_model or base_model