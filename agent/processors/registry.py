from __future__ import annotations

from ..config import DomainConfig
from .analysis import AnalysisProcessor
from .base import Processor
from .coding import CodingProcessor
from .debugging import DebuggingProcessor
from .direct import DirectAnswerProcessor
from .teaching import TeachingProcessor

PROCESSOR_CLASSES = {
    "direct": DirectAnswerProcessor,
    "teaching": TeachingProcessor,
    "debugging": DebuggingProcessor,
    "analysis": AnalysisProcessor,
    "coding": CodingProcessor,
}


def build_registry(domain: DomainConfig) -> dict[str, Processor]:
    return {
        sid: cls(domain.prompts[sid], domain.name, domain.description)
        for sid, cls in PROCESSOR_CLASSES.items()
    }
