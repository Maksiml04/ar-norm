from __future__ import annotations
from enum import Enum, StrEnum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

class ChunkType(str, Enum):
    SECTION_HEADER  = "SECTION_HEADER"
    TABLE_TITLE     = "TABLE_TITLE"
    TABLE_BLOCK     = "TABLE_BLOCK"
    FIGURE_CAPTION  = "FIGURE_CAPTION"
    TEXT_BODY       = "TEXT_BODY"
    LIST_ITEM       = "LIST_ITEM"
    NOTE            = "NOTE"
    FORMULA         = "FORMULA"
    APPENDIX_HEADER = "APPENDIX_HEADER"

class DocumentChunk(BaseModel):
    chunk_id:   str
    chunk_type: ChunkType
    text:       str
    location:   dict[str, Any] = Field(default_factory=dict)
    metadata:   dict[str, Any] = Field(default_factory=dict)
    is_bold:     Optional[bool]  = None
    is_centered: Optional[bool]  = None
    font_size:   Optional[float] = None

class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR    = "major"
    MINOR    = "minor"

class Violation(BaseModel):
    rule_id:        str
    violation_type: str
    explanation:    str
    severity:       Severity = Severity.MINOR
    reasoning:      Optional[str] = None

# Схема для Structured Outputs от LLM (Гарантирует формат ответа)
class LLMOutput(BaseModel):
    has_violation: bool
    violations: list[Violation] = Field(default_factory=list)

class ValidatorType(str, Enum):
    RULE_BASED = "rule_based"
    LLM        = "llm"
    MERGED     = "merged"
    HYBRID = "hybrid"


class ValidationResult(BaseModel):
    chunk_id: str
    validator: "ValidatorType"  # ← Кавычки для отложенного разрешения, если Enum ниже
    has_violation: bool
    violations: List["Violation"]  # ← Если Violation определён ниже
    confidence: float
    location: Optional[Dict[str, Any]] = None  # ← Теперь работает, т.к. Dict/Any импортированы

    model_config = {"arbitrary_types_allowed": True}  # Опционально, если нужны сложные типы

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        merged_violations = self.violations + [
            v for v in other.violations
            if v.rule_id not in {x.rule_id for x in self.violations}
        ]
        return ValidationResult(
            chunk_id      = self.chunk_id,
            validator     = ValidatorType.MERGED,
            has_violation = bool(merged_violations),
            violations    = merged_violations,
            confidence    = max(self.confidence, other.confidence),
        )

ValidationResult.model_rebuild()
Violation.model_rebuild()