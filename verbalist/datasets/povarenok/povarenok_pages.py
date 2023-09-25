import requests
from bs4 import BeautifulSoup
import pandas as pd
import markdownify
import re
from tqdm import tqdm
from datasets import Dataset
import concurrent.futures
import os
from datasets import load_dataset
import itertools


class PagesParser:
    def __init__(
        self,
        items=None,
    ) -> None:
        self.items = items
        self.dataset = []

    def parse_func(self, item):
        # url = "https://www.povarenok.ru/recipes/show/130312/"
        url = item["link"]
        response = requests.get(url=url)
        soup = BeautifulSoup(response.text, features="lxml")

        title_receipt = ""
        steps = []
        full_receipt_text = ""

        try:
            title_receipt = soup.find_all("h2")[2].text
            steps = soup.find_all("li", class_="cooking-bl")
            steps = [item.text.strip() for item in steps]
            full_receipt_text = [f"{i+1}. {item}" for i, item in enumerate(steps)]
            full_receipt_text = "\n".join(full_receipt_text)
            full_receipt_text = f"{title_receipt}\n{full_receipt_text}"
        except:
            title_receipt = ""
            steps = []
            full_receipt_text = ""

        return {
            "full_receipt_text": full_receipt_text,
            "steps": steps,
            "title_receipt": title_receipt,
            **item,
        }

    def parse_parallel(self):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            texts = tqdm(
                executor.map(
                    self.parse_func,
                    self.items,
                    chunksize=1,
                )
            )
            texts = list(texts)
        self.dataset = texts


if __name__ == "__main__":
    dataset = load_dataset("dim/povarenok_links")
    dataset = dataset["train"]
    items = dataset.to_list()
    parser = PagesParser(items=items)
    parser.parse_parallel()
    # print(parser.dataset)
    dataset = Dataset.from_list(parser.dataset)
    dataset.push_to_hub("dim/povarenok")
