from __future__ import annotations

from .base import Processor


class TeachingProcessor(Processor):
    strategy_id = "teaching"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Concept\n"
            "- Why it is designed this way\n"
            "- How it works\n"
            "- Concrete example\n"
            "- Common misconceptions\n"
            "- Summary"
        )
