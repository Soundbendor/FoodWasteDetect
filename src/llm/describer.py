import json
import subprocess
from collections import defaultdict

from ollama import ChatResponse, chat

from datasets.dataset import Dataset


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
            subprocess.call("start_deepseek.sh", shell=True)

    def generate_descriptors(self, ds: Dataset) -> str:
        """Given a dataset, provides a list of descriptions of each class of that dataset"""
        ds_descriptors = defaultdict(list)
        for item in self._load_dataset(ds):
            response: ChatResponse = chat(
                model="deepseek-r1:32b",
                messages=[
                    {
                        "role": "user",
                        "content": f"Can you provide 10 sentences describing the visual characteristics of the food {item}?",
                    },
                ],
            )
            for line in response.message.content.splitlines():
                if len(line) > 3:
                    if line[0].isdigit():
                        descriptor = line.partition(" ")[2]
                        ds_descriptors[item].append(descriptor)

        with open("descriptor_dictionary.json", "w", encoding="utf-8") as f:
            json.dump(ds_descriptors, f, ensure_ascii=False, indent=4)

        return "descriptor_dictionary.json"
