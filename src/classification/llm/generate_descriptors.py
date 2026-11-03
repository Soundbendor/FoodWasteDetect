"""Generate visual food descriptions through a supplied Ollama client."""

from collections.abc import Sequence

from ollama import ChatResponse, Client
from pydantic import BaseModel
from tqdm import tqdm


def descriptor_prompt(cleaned_item: str) -> dict[str, str]:
    """Build a visual-description request using the supplied food label verbatim."""
    return {
        "role": "user",
        "content": (
            f"Can you provide a sentence describing the food {cleaned_item}? "
            "Please be as descriptive as possible, focusing on the visual "
            f"characteristics of {cleaned_item} and explicitly mentioning "
            f"{cleaned_item} in your description."
        ),
    }


# Keep the response schema unchanged: Pydantic includes class docstrings in JSON Schema.
class FoodDescriptor(BaseModel):
    food_class: str
    description: str


class DescriberLLM:
    """Generate food descriptions through a caller-owned Ollama client."""

    def __init__(self, client: Client, model: str) -> None:
        """Use an existing client without starting or stopping a server."""
        self.client = client
        self.model = model

    def describe_class(self, label: str) -> FoodDescriptor:
        """Generate one description, preserving the supplied label and stripping asterisks."""
        return self._describe_class(label, descriptor_prompt(label))

    def generate_descriptions(
        self,
        classes: Sequence[str],
        count: int,
    ) -> dict[str, list[str]]:
        """Generate ``count`` descriptions per class, preserving request order.

        Progress advances after each returned request. Request and validation
        errors propagate without retries or a partial result.

        Returns:
            A mapping from each supplied class label to its description list.

        Raises:
            ValueError: A class has no descriptions or a description is blank.
        """
        descriptions = {}
        with tqdm(
            total=len(classes) * count, desc="Descriptions", unit="description", mininterval=5,
        ) as progress:
            for label in classes:
                descriptions[label] = []
                for _ in range(count):
                    description = self.describe_class(label).description
                    progress.update(1)
                    descriptions[label].append(description)
                if not descriptions[label] or any(not text.strip() for text in descriptions[label]):
                    raise ValueError(f"Empty generated description for {label}")
        return descriptions

    def _describe_class(self, label: str, prompt: dict[str, str]) -> FoodDescriptor:
        """Request and validate one description, retaining the supplied class label."""
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=[prompt],
            format=FoodDescriptor.model_json_schema(),
        )
        output = FoodDescriptor.model_validate_json(response.message.content or "")
        output.food_class = label
        output.description = output.description.replace("*", "")
        return output
