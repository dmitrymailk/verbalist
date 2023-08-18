from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.firefox.options import Options

import requests
from tqdm import tqdm

import multiprocessing as mp

from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor

import pandas as pd
import json


class OpenReviewsParser:
    def __init__(self) -> None:
        path_to_driver = "./geckodriver.exe"
        self.service = Service(path_to_driver)
        self.driver_options = Options()
        self.timeout = 10
        self.dataset = []

    def get_all_page_links(self, workshop_url):
        paper_urls = []
        try:
            driver = webdriver.Firefox(
                # service=self.service,
                options=self.driver_options,
            )
            driver.refresh()
            driver.get(url=workshop_url)
            WebDriverWait(driver, timeout=self.timeout).until(
                lambda d: d.find_element(
                    by=By.CLASS_NAME,
                    value="note",
                ),
            )

            all_links = driver.find_elements(by=By.CLASS_NAME, value="note")
            for link in tqdm(all_links):
                link = link.find_element(by=By.TAG_NAME, value="a").get_property("href")
                paper_id = str(link).split("id=")[1]
                review_url = f"https://api.openreview.net/notes?forum={paper_id}"
                reviews = requests.get(review_url).json()
                reviews = [
                    item
                    for item in reviews["notes"]
                    if not "TL;DR" in item["content"] and "review" in item["content"]
                ]
                if len(reviews) > 0:
                    paper_urls.append(
                        {
                            "paper_url": link,
                            "paper_id": paper_id,
                            "reviews": reviews,
                        }
                    )

            driver.close()
        except Exception as e:
            print(e)
            # driver.close()

        return paper_urls

    def get_all_subjects(self, root_url):
        subject_urls = []
        try:
            driver = webdriver.Firefox(
                # service=self.service,
                options=self.driver_options,
            )
            driver.refresh()
            driver.get(url=root_url)

            WebDriverWait(driver, timeout=self.timeout).until(
                lambda d: d.find_element(
                    by=By.CLASS_NAME,
                    value="list-unstyled.venues-list",
                ),
            )

            links_list = driver.find_element(
                by=By.CLASS_NAME, value="list-unstyled.venues-list"
            )

            all_links = links_list.find_elements(by=By.TAG_NAME, value="a")

            for link in tqdm(all_links):
                link = link.get_property("href")
                subject_urls.append(link)

            driver.close()
        except Exception as e:
            print(e)

        return subject_urls

    def parse_all_conferences(self):
        conference_urls = [
            "https://openreview.net/group?id=aclweb.org/ACL/2022/Workshop"
        ]

        for conference_url in tqdm(conference_urls):
            subject_urls = self.get_all_subjects(
                root_url=conference_url,
            )

            for workshop_url in tqdm(subject_urls):
                try:
                    papers_with_reviews = self.get_all_page_links(
                        workshop_url=workshop_url,
                    )
                    self.dataset.extend(papers_with_reviews)
                except Exception as e:
                    print(e)

    def parse_all_conferences_parallel(self):
        conference_urls = [
            "https://openreview.net/group?id=aclweb.org/ACL/2022/Workshop"
        ]

        for conference_url in tqdm(conference_urls):
            subject_urls = self.get_all_subjects(
                root_url=conference_url,
            )
        with ThreadPoolExecutor(max_workers=30) as worker:
            future_papers_with_reviews = [
                worker.submit(self.get_all_page_links, link) for link in subject_urls
            ]

            for future_papers_with_review in future_papers_with_reviews:
                try:
                    papers_with_reviews = future_papers_with_review.result()
                    # self.dataset.extend(papers_with_reviews)
                    self.dataset.extend(papers_with_reviews)
                except Exception as e:
                    print(e)
                    pass


if __name__ == "__main__":
    parser = OpenReviewsParser()
    # parser.get_all_page_links()
    # parser.get_all_subjects()
    # parser.parse_all_conferences()
    parser.parse_all_conferences_parallel()
    print(parser.dataset)
    with open("./dataset.csv")
    # pd.DataFrame(data=parser.dataset).to_csv("./dataset.csv", index=False)
