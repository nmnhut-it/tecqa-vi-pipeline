"""
Verbatim prompts from paper Appendix F.3.
"""
from .stage1 import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    MAIN_ENTITY_PROMPT,
)
from .stage3 import QA_REASONING_PROMPT

__all__ = [
    "ENTITY_EXTRACTION_PROMPT",
    "RELATION_EXTRACTION_PROMPT",
    "MAIN_ENTITY_PROMPT",
    "QA_REASONING_PROMPT",
]
