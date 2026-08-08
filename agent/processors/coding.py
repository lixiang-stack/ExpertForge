from __future__ import annotations

from .base import Processor


class CodingProcessor(Processor):
    strategy_id = "coding"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Approach\n"
            "- Code with inline explanation\n"
            "- Key considerations\n"
            "- Best practices"
        )
