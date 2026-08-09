from __future__ import annotations

from .base import Processor


class CodeSnippetProcessor(Processor):
    strategy_id = "code_snippet"

    @property
    def structure(self) -> str:
        return (
            "Answer in this structure:\n"
            "- Approach\n"
            "- Code snippet\n"
            "- Key points and caveats\n"
            "- How to extend or adapt it"
        )
