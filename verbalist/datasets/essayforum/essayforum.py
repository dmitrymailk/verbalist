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


class EssayForumMessagesParser:
    def __init__(self, urls) -> None:
        self.urls = urls

        self.dataset = []

    def parse(self):
        for url_dict in tqdm(self.urls):
            url = url_dict["url"]
            forum_type = url_dict["forum_type"]

            response = requests.get(url=url)
            soup = BeautifulSoup(response.text)
            all_messages = soup.find_all("article")
            dataset_item = []
            for pos, message in enumerate(all_messages):
                author = message.find("div", "fl").find_all("a")
                if len(author) > 0:
                    author = author[0].find("b").text
                else:
                    author = message.find("div", "fl").text.replace("/", "").strip()
                # print(author)

                number = all_messages[0].find("div", "fr").find("a").text
                date = (
                    all_messages[0].find("div", "fr").text.replace(number, "").strip()
                )
                # print(date)

                message = str(message.find("div", "pTx"))
                message = markdownify.markdownify(message, heading_style="ATX")
                message = re.sub("\!\[\]\(\w+.*\)", "", message).strip()
                # print(message)
                # print("=" * 100)
                # print("=" * 100)
                dataset_item.append(
                    {
                        "message": message,
                        "author": author,
                        "date": date,
                        "position": pos,
                        "url": url,
                        "forum_type": forum_type,
                    }
                )

            self.dataset.append(dataset_item)
        self.dataset = list(itertools.chain(*self.dataset))

    def parse_parallel(self):
        for url_dict in tqdm(self.urls):
            pass

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            texts = tqdm(
                executor.map(
                    self.parse_func,
                    self.urls,
                    chunksize=10,
                )
            )
            texts = list(texts)
        self.dataset = list(itertools.chain(*texts))

    def parse_func(self, url_dict):
        url = url_dict["url"]
        forum_type = url_dict["forum_type"]

        response = requests.get(url=url)
        soup = BeautifulSoup(response.text)
        all_messages = soup.find_all("article")
        dataset_item = []
        for pos, message in enumerate(all_messages):
            author = message.find("div", "fl").find_all("a")
            if len(author) > 0:
                author = author[0].find("b").text
            else:
                author = message.find("div", "fl").text.replace("/", "").strip()
            # print(author)

            number = all_messages[0].find("div", "fr").find("a").text
            date = all_messages[0].find("div", "fr").text.replace(number, "").strip()
            # print(date)

            message = str(message.find("div", "pTx"))
            message = markdownify.markdownify(message, heading_style="ATX")
            message = re.sub("\!\[\]\(\w+.*\)", "", message).strip()
            # print(message)
            # print("=" * 100)
            # print("=" * 100)
            dataset_item.append(
                {
                    "message": message,
                    "author": author,
                    "date": date,
                    "position": pos,
                    "url": url,
                    "forum_type": forum_type,
                }
            )

        return dataset_item

class EssayForumLinksParser:
    def __init__(self, urls) -> None:
        self.base_urls = urls

        self.dataset = []

    def parse(self):
        for base_url in self.base_urls:
            response = requests.get(url=base_url)
            soup = BeautifulSoup(response.text)

            total_pages = soup.find_all("a", class_="navCell")[-1].text
            total_pages = int(total_pages)

            forum_type = base_url.replace("https://", "")
            forum_type = forum_type.replace("/", "_")

            for pos in tqdm(range(1, total_pages + 1)):
                url = f"{base_url}/{pos}"
                response = requests.get(url=url)
                soup = BeautifulSoup(response.text)
                all_links = soup.find_all("a", class_="txtNr")

                for link in all_links:
                    self.dataset.append({"forum_type": forum_type, "url": link["href"]})

if __name__ == "__main__":
    links = pd.read_csv("./essayforum_writing_links.csv")
    links = links.to_dict("records")
    links = links[500:10500]

    parser = EssayForumMessagesParser(urls=links)
    # parser.parse()
    parser.parse_parallel()
    dataset = Dataset.from_list(parser.dataset)
    dataset.push_to_hub('dim/essayforum_writing_10k')
