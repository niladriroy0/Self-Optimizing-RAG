import threading
from evaluation.rag_evaluator import evaluate_rag
from optimization.experiment_db import log_experiment
from optimization.optimizer import adapt_config   # ✅ NEW
from control_plane.config_manager import config_manager  # ✅ NEW
from cache.chroma_memory_store import store_memory


def run_evaluation_async(question, answer, context, config, pipeline_confidence):

    def task():
        try:
            evaluation = evaluate_rag(question, answer, context)

            relevance = float(evaluation.get("answer_relevance", 0))
            faithfulness = float(evaluation.get("faithfulness", 0))

            # ----------------------------------
            # 🔥 STORE IN SQLITE
            # ----------------------------------
            log_experiment(question, config, relevance, faithfulness)

            # ----------------------------------
            # 🔥 COMPUTE CONFIDENCE
            # ----------------------------------
            confidence = (pipeline_confidence * 0.7) + ((relevance + faithfulness)/2 * 0.3)

            # ----------------------------------
            # 🔥 STORE MEMORY
            # ----------------------------------
            store_memory(question, answer, confidence)

            # ----------------------------------
            # 🔥 REAL-TIME ADAPTATION (NEW)
            # ----------------------------------
            new_config = adapt_config(config, relevance, faithfulness)

            if new_config:
                print("⚙️ Updating config from feedback:", new_config)
                config_manager.update_config(new_config)

        except Exception as e:
            print("❌ Evaluation Worker Error:", e)

    thread = threading.Thread(target=task)
    thread.start()

    thread.join(timeout=0.1)