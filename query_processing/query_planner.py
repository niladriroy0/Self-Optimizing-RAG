# query_processing/query_planner.py

from typing import List

from query_processing.query_decomposer import decompose_query_llm
from control_plane.config_manager import config_manager


def plan_query(query: str, analysis: dict) -> List[str]:
    """
    Decides:
    - single query
    - or decomposed queries
    """

    enable_multi_hop = config_manager.get_param("enable_multi_hop", True)
    enable_decomposition = config_manager.get_param("enable_decomposition", True)

    # ----------------------------------
    # MULTI-HOP DECISION
    # ----------------------------------

    if (
        enable_multi_hop
        and enable_decomposition
        and analysis.get("is_multi_hop", False)
    ):
        return decompose_query_llm(query)

    # ----------------------------------
    # SINGLE QUERY
    # ----------------------------------

    return [query]