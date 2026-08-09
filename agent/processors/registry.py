from __future__ import annotations

from ..config import DomainConfig
from .analysis import AnalysisProcessor
from .base import Processor
from .code_snippet import CodeSnippetProcessor
from .debugging import DebuggingProcessor
from .direct import DirectAnswerProcessor
from .teaching import TeachingProcessor

PROCESSOR_CLASSES = {
    "direct": DirectAnswerProcessor,
    "teaching": TeachingProcessor,
    "debugging": DebuggingProcessor,
    "analysis": AnalysisProcessor,
    "code_snippet": CodeSnippetProcessor,
}


def build_registry(domain: DomainConfig) -> dict[str, Processor]:
    return {
        sid: cls(domain.prompts[sid], domain.name, domain.description)
        for sid, cls in PROCESSOR_CLASSES.items()
        if sid in domain.prompts
    }
