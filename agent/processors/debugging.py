from __future__ import annotations

from .base import Processor


class DebuggingProcessor(Processor):
    strategy_id = "debugging"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Problem analysis\n"
            "- Possible causes\n"
            "- Verification steps\n"
            "- Fix suggestions\n"
            "- Best practices"
        )
