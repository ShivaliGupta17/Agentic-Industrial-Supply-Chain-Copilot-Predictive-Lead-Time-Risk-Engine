"""
LangGraph State Definition for Supply Chain Research Agent.
Tracks document content, extracted bottleneck signals, cyclic validation state, and database staging.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class BottleneckSignal(TypedDict):
    supplier_name: str
    component_category: str
    reported_delay_weeks: float
    root_cause: str
    commodity_driver: str
    confidence_score: float


class AgentState(TypedDict):
    report_name: str
    raw_text: str
    extracted_signals: List[Dict[str, Any]]
    validation_passed: bool
    validation_errors: List[str]
    retry_count: int
    db_records_inserted: int
    execution_log: List[str]
