# livecode/lcb_runner/lm_styles.py
#
# Minimal copy: LMStyle enum and LanguageModel dataclass only.
# Full LanguageModelList omitted — not needed for HEART/livecode.

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class LMStyle(Enum):
    OpenAIChat = "OpenAIChat"
    OpenAIReasonPreview = "OpenAIReasonPreview"
    OpenAIReason = "OpenAIReason"

    Claude = "Claude"
    Claude3 = "Claude3"
    Claude3Thinking = "Claude3Thinking"

    Gemini = "Gemini"
    GeminiThinking = "GeminiThinking"
    Gemini3 = "Gemini3"
    Grok = "Grok"

    MistralWeb = "MistralWeb"
    CohereCommand = "CohereCommand"

    DataBricks = "DataBricks"
    DeepSeekAPI = "DeepSeekAPI"

    GenericBase = "GenericBase"

    DeepSeekCodeInstruct = "DeepSeekCodeInstruct"
    CodeLLaMaInstruct = "CodeLLaMaInstruct"
    StarCoderInstruct = "StarCoderInstruct"
    CodeQwenInstruct = "CodeQwenInstruct"
    QwQ = "QwQ"
    LLaMa3 = "LLaMa3"
    DeepSeekR1 = "DeepSeekR1"

    TogetherAI = "TogetherAI"


@dataclass
class LanguageModel:
    model_name: str
    model_repr: str
    model_style: LMStyle
    release_date: datetime | None
    link: str | None = None

    def __hash__(self) -> int:
        return hash(self.model_name)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_repr": self.model_repr,
            "model_style": self.model_style.value,
            "release_date": int(self.release_date.timestamp() * 1000),
            "link": self.link,
        }
