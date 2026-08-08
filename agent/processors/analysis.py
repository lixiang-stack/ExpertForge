from __future__ import annotations

from .base import Processor


class AnalysisProcessor(Processor):
    strategy_id = "analysis"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Comparison dimensions\n"
            "- Key differences\n"
            "- Trade-offs\n"
            "- Recommendation"
        )
