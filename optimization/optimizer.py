import random
from optimization.experiment_db import get_best_config
from control_plane.config_manager import config_manager


DEFAULT_CONFIGS = [
    {"top_k": 2, "chunk_size": 200},
    {"top_k": 3, "chunk_size": 400},
    {"top_k": 4, "chunk_size": 600},
]


def choose_config():
    best = get_best_config()

    if best:
        return best

    return random.choice(DEFAULT_CONFIGS)


def adapt_config(config, relevance, faithfulness):

    new_config = config.copy()
    changed = False

    # ----------------------------------
    # 🔥 FAITHFULNESS FIX
    # ----------------------------------
    if faithfulness < 0.5:
        new_top_k = min(config["top_k"] + 1, 8)
        if new_top_k != config["top_k"]:
            new_config["top_k"] = new_top_k
            changed = True

    # ----------------------------------
    # 🔥 RELEVANCE FIX
    # ----------------------------------
    if relevance < 0.5:
        new_chunk = min(config["chunk_size"] + 200, 1200)
        if new_chunk != config["chunk_size"]:
            new_config["chunk_size"] = new_chunk
            changed = True

    # ----------------------------------
    # 🔥 ONLY UPDATE IF CHANGED
    # ----------------------------------
    if changed:
        config_manager.update_config(new_config)
        return new_config

    return None