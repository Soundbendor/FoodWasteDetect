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
        self.model = "deepseek-r1:32b"

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
                    model="deepseek-r1:32b",
                    messages=[
                        {
                            "role": "user",
                            "content": f"Can you provide a sentence describing the food {item}? Please be as descriptive as possible, focusing on the visual characteristics of the food.",
                        },
                    ],
                    format=FoodDescriptor.model_json_schema(),
                )
                output = FoodDescriptor.model_validate_json(response.message.content)
                print(output)
                # ds_descriptors[item].append(response.message.content)
                # for line in response.message.content.splitlines():
                # if len(line) > 3:
                # if line[0].isdigit():
                # descriptor = line.partition(" ")[2]
                # ds_descriptors[item].append(descriptor)

        with open("assets/descriptor_dictionary.json", "w", encoding="utf-8") as f:
            json.dump(ds_descriptors, f, ensure_ascii=False, indent=4)

        return "assets/descriptor_dictionary.json"
