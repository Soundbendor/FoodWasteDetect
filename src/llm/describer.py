import json
import subprocess
from collections import defaultdict

from ollama import ChatResponse, chat
from pydantic import BaseModel

from datasets.dataset import Dataset


class FoodDescriptor(BaseModel):
    food_class: str
    description: str


class DescriberLLM:
    def __init__(self):
        self.model = "gemma3:27b"

    def _load_dataset(self, ds: Dataset):
        return ds.get_class_list()

    def start_server(self):
        try:
            chat(
                model=self.model,
                messages=[{"role": "user", "content": "Hey, you alive?"}],
            )
        except ConnectionError:
            # Run online script
            print("Error handler!")
            process = subprocess.Popen(
                "src/llm/start_deepseek.sh",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out, err = process.communicate()
            print(err)
            print(out)
            # subprocess.call("bash llm/start_deepseek.sh", shell=True)

    def generate_descriptors(self, ds: Dataset) -> str:
        """Given a dataset, provides a list of descriptions of each class of that dataset"""
        ds_descriptors = defaultdict(list)
        for item in self._load_dataset(ds)[:5]:
            for _ in range(5):
                response: ChatResponse = chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Can you provide a sentence describing the food {item}? Please be as descriptive as possible, focusing on the visual characteristics of the {item} and explicitly mention {item} in your description.",
                        },
                    ],
                    format=FoodDescriptor.model_json_schema(),
                )
                output = FoodDescriptor.model_validate_json(response.message.content)
                # Make sure the food class always matches our class label
                output.food_class = item
                # Remove emphasis
                output.description = output.description.replace("*", "")
                print(output)
                ds_descriptors[item].append(output.description)

        with open("assets/descriptor_dictionary.json", "w", encoding="utf-8") as f:
            json.dump(ds_descriptors, f, ensure_ascii=False, indent=4)

        return "assets/descriptor_dictionary.json"
