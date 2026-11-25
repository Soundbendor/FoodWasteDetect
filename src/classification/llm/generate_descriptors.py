"""Generate visual food descriptions through Ollama or Gemini on Vertex AI."""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from google import genai
from google.genai import types
from ollama import ChatResponse, Client
from pydantic import BaseModel
from tqdm import tqdm

TEMPLATE_VERSION = 1
_QUESTIONS = {
    "overall": "What does {label} typically look like in a food photograph?",
    "shape": "What shape or arrangement of pieces helps someone recognize {label}?",
    "surface": "What visible surface pattern, texture, or color variation helps identify {label}?",
    "distinguishing": "Which visible feature of {label} is especially useful for telling it apart from similar foods?",
    "variation": "Describe another common way {label} appears when photographed whole, cut, or served. Choose one form appropriate to the food.",
}
TEMPLATES = {
    key: question + " Answer in one short sentence naming {label} and describing visible features."
    for key, question in _QUESTIONS.items()
}


def template_order(count: int) -> list[str]:
    """Select one template per caption in generation order."""
    if count <= 0:
        raise ValueError("description_count must be positive")
    keys = list(TEMPLATES)
    return [keys[i % len(keys)] for i in range(count)]


def descriptor_prompt(label: str, template_id: str = "overall") -> dict[str, str]:
    return {"role": "user", "content": TEMPLATES[template_id].format(label=label)}


class FoodDescriptor(BaseModel):
    food_class: str
    description: str


class CaptionGenerator(ABC):
    """Apply the shared caption prompts, validation and replacement policy."""

    @abstractmethod
    def _request(self, message: dict[str, str]) -> str:
        """Return a structured descriptor as JSON."""

    def describe_class(
        self, label: str, template_id: str = "overall", replacement: bool = False
    ) -> FoodDescriptor:
        message = descriptor_prompt(label, template_id)
        if replacement:
            message["content"] += " Write a different caption and include the exact food name."
        output = FoodDescriptor.model_validate_json(self._request(message))
        output.food_class = label
        output.description = output.description.replace("*", "").strip()
        return output

    def generate_descriptions(
        self,
        classes: Sequence[str],
        count: int,
        *,
        on_class: Callable[[str, list[str]], None] | None = None,
    ) -> dict[str, list[str]]:
        """Request each template separately and checkpoint only complete classes."""
        order = template_order(count)
        descriptions = {}
        with tqdm(
            total=len(classes) * count, desc="Descriptions", unit="description", mininterval=5
        ) as progress:
            for label in classes:
                captions = []
                seen = set()
                for template_id in order:
                    for replacement in (False, True):
                        text = self.describe_class(label, template_id, replacement).description
                        key = " ".join(text.casefold().split())
                        if key not in seen and re.search(
                            r"(?<!\w)" + re.escape(label.casefold()) + r"(?!\w)", key
                        ):
                            break
                    else:
                        raise ValueError(
                            f"Invalid generated description for {label} ({template_id})"
                        )
                    captions.append(text)
                    seen.add(key)
                    progress.update(1)
                if on_class:
                    on_class(label, captions)
                descriptions[label] = captions
        return descriptions


class DescriberLLM(CaptionGenerator):
    """Generate food descriptions through a caller-owned Ollama client."""

    def __init__(self, client: Client, model: str) -> None:
        self.client = client
        self.model = model

    def _request(self, message: dict[str, str]) -> str:
        response: ChatResponse = self.client.chat(
            model=self.model, messages=[message], format=FoodDescriptor.model_json_schema()
        )
        return response.message.content or ""


class GeminiDescriber(CaptionGenerator):
    """Generate food descriptions through a caller-owned Google Gen AI client."""

    def __init__(self, client: genai.Client, model: str) -> None:
        self.client = client
        self.model = model

    def _request(self, message: dict[str, str]) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=message["content"],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=FoodDescriptor
            ),
        )
        return response.text or ""
