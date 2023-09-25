import requests
from bs4 import BeautifulSoup
import pandas as pd
import markdownify
import re
from tqdm import tqdm
from datasets import Dataset
import concurrent.futures
import os

import itertools


class LinksParser:
    def __init__(
        self,
    ) -> None:
        self.domain = "https://www.povarenok.ru/recipes"
        self.dataset = []

    def parse_func(self, page_num):
        url = f"{self.domain}/~{page_num}/?sort=rating&order=desc"
        response = requests.get(url=url)
        soup = BeautifulSoup(response.text)
        items = soup.find_all("article", class_="item-bl")
        dataset = []

        for card in items:
            title = card.find("h2").text.strip()
            ingridients = [
                item.text for item in card.find("span", class_="list").find_all("a")
            ]
            views = int(card.find("span", class_="i-views").text)
            likes = int(card.find("span", class_="i-likes").text)
            ups = int(card.find("span", class_="i-up").text)
            link = card.find("h2").find("a")["href"]

            dataset.append(
                {
                    "title": title,
                    "ingridients": ingridients,
                    "views": views,
                    "likes": likes,
                    "ups": ups,
                    "link": link,
                }
            )
        return dataset

    def parse_parallel(self):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            texts = tqdm(
                executor.map(
                    self.parse_func,
                    list(range(1, 3101)),
                    chunksize=1,
                )
            )
            texts = list(texts)
        self.dataset = list(itertools.chain(*texts))


if __name__ == "__main__":
    parser = LinksParser()
    parser.parse_parallel()
    # print(parser.dataset)
    dataset = Dataset.from_list(parser.dataset)
    dataset.push_to_hub("dim/povarenok_links")
