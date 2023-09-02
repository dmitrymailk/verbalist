from datasets import disable_caching

disable_caching()

import random
import json
from typing import Optional
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from src.util.chat import Conversation, ConversationVerbalist
import hashlib
import concurrent.futures
from joblib import Parallel, delayed
import os


from datasets import load_dataset


class ChatDatasetSaiga(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        sample_rate: float = 1.0,
        only_target_loss: bool = True,
        add_global_bos: bool = False,
        add_global_eos: bool = False,
        dataset_type: str = "train",
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.is_printed = False
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.dataset_type = dataset_type

        self.records = []

        filename = f"{tokenizer.name_or_path}{sample_rate}{templates_path}{max_tokens_count}".encode()
        print(filename)
        filename = hashlib.sha256(filename).hexdigest()
        filename = f"./models/temp/{filename}_{dataset_type}.bin"
        print(filename)

        if os.path.isfile(filename):
            self.records = torch.load(filename)
        else:
            for record in tqdm(original_records):
                if random.random() > self.sample_rate:
                    continue
                tensors = self.convert_record(record)
                if tensors is None:
                    continue
                self.records.append(tensors)
            torch.save(self.records, filename)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, text):
        return self.tokenizer(
            text, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]

    def convert_record(self, record):
        conversation = Conversation.from_template(self.templates_path)
        conversation.expand(record["messages"])
        full_text = conversation.get_prompt(self.tokenizer, add_suffix=False)

        if not self.is_printed:
            print("Prompt example")
            print(full_text)
            self.is_printed = True

        input_ids = self.get_tokens(full_text)
        if self.add_global_bos:
            input_ids.insert(0, self.tokenizer.bos_token_id)
        input_ids = input_ids[: self.max_tokens_count - 1]
        if self.add_global_eos or input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids.append(self.tokenizer.eos_token_id)
        actual_length = len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.only_target_loss:
            start_token_id = conversation.get_start_token_id()
            end_token_id = conversation.get_end_token_id()
            bot_token_id = conversation.get_bot_token_id()

            spans = []
            cur_start_idx = -1
            cur_end_idx = -1
            cur_is_bot = False

            input_ids = input_ids.tolist()
            while True:
                try:
                    cur_start_idx = input_ids.index(start_token_id, cur_start_idx + 1)
                    cur_end_idx = input_ids.index(end_token_id, cur_start_idx + 1) + 1
                    cur_is_bot = (
                        input_ids[cur_start_idx:cur_end_idx].count(bot_token_id) >= 1
                    )
                    if not cur_is_bot:
                        spans.append((cur_start_idx, cur_end_idx))
                except ValueError:
                    break
            for start_idx, end_idx in spans:
                start_idx = max(0, start_idx)
                end_idx = min(len(input_ids), end_idx)
                labels[start_idx:end_idx] = -100

            if (labels == start_token_id).sum() == 0:
                return None

            assert (labels == start_token_id).sum() == (labels == end_token_id).sum()
            assert (labels == bot_token_id).sum() >= (labels == start_token_id).sum()

        input_ids = torch.LongTensor(input_ids)
        assert input_ids.size(0) == labels.size(0) == attention_mask.size(0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class ChatDatasetVerbalist(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        tokenizer: AutoTokenizer,
        max_tokens_count: int,
        templates_path: str,
        test_size: int,
        only_target_loss: bool = True,
        add_global_bos: bool = False,
        add_global_eos: bool = False,
        dataset_type: str = "train",
        status: str = "ok",
    ):
        self.templates_path = templates_path
        self.original_records = original_records
        self.tokenizer = tokenizer
        self.max_tokens_count = max_tokens_count
        self.only_target_loss = only_target_loss
        self.is_printed = False
        self.add_global_bos = add_global_bos
        self.add_global_eos = add_global_eos
        self.dataset_type = dataset_type
        self.status = status

        self.records = []

        # filename = f"{tokenizer.name_or_path}{templates_path}{max_tokens_count}{test_size}{status}".encode()
        # print(filename)
        # filename = hashlib.sha256(filename).hexdigest()
        # filename = f"./models/temp/{filename}_{dataset_type}.bin"
        # print(filename)

        # if os.path.isfile(filename):
        #     self.records = torch.load(filename)
        # else:
        for record in tqdm(original_records):
            tensors = self.convert_record(record)
            if tensors is None:
                print(record)
                assert False, "Something wrong with you chat record"
            self.records.append(tensors)
            # torch.save(self.records, filename)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get_tokens(self, text):
        return self.tokenizer(
            text, add_special_tokens=False, padding=False, truncation=False
        )["input_ids"]

    def convert_record(self, record):
        conversation = ConversationVerbalist.from_template(self.templates_path)
        conversation.expand(record["conversation_text"])
        full_text = conversation.get_prompt(self.tokenizer, add_suffix=False)

        if not self.is_printed:
            print("Prompt example")
            print(full_text)
            self.is_printed = True

        input_ids = self.get_tokens(full_text)
        input_ids.insert(0, self.tokenizer.bos_token_id)
        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.only_target_loss:
            start_token_id = conversation.get_start_token_id()
            end_token_id = conversation.get_end_token_id()
            bot_token_id = conversation.get_bot_token_id()

            spans = []
            cur_start_idx = -1
            cur_end_idx = -1
            cur_is_bot = False

            input_ids = input_ids.tolist()
            while True:
                try:
                    cur_start_idx = input_ids.index(start_token_id, cur_start_idx + 1)
                    cur_end_idx = input_ids.index(end_token_id, cur_start_idx + 1) + 1
                    cur_is_bot = (
                        input_ids[cur_start_idx:cur_end_idx].count(bot_token_id) >= 1
                    )
                    if not cur_is_bot:
                        spans.append((cur_start_idx, cur_end_idx))
                except ValueError:
                    break

            for start_idx, end_idx in spans:
                start_idx = max(0, start_idx)
                end_idx = min(len(input_ids), end_idx)
                labels[start_idx:end_idx] = -100

            if (labels == start_token_id).sum() == 0:
                print("=" * 100)
                print("=" * 100)
                print("=" * 100)
                print(record)
                print("=" * 100)
                print(full_text)
                print("=" * 100)
                raise "Something wrong with you code"

            assert (labels == start_token_id).sum() == (labels == end_token_id).sum()
            assert (labels == bot_token_id).sum() >= (labels == start_token_id).sum()

            input_ids = torch.LongTensor(input_ids)
            return {
                "input_ids": input_ids[: self.max_tokens_count],
                "attention_mask": attention_mask[: self.max_tokens_count],
                "labels": labels[: self.max_tokens_count],
            }


class ChatDatasetVerbalistUnion(Dataset):
    def __init__(
        self,
        dataset_configs: list[dict],
        tokenizer: AutoTokenizer,
        templates_path: str,
        max_tokens_count: int = 2048,
    ) -> None:
        self.dataset_configs = dataset_configs
        self.chat_datasets = []
        self.tokenizer = tokenizer
        self.templates_path = templates_path
        self.max_tokens_count = max_tokens_count

        self.concat_dataset_train = []
        self.concat_dataset_test = []

        self.conversation_field = "conversation_text"

        self.get_dataset_parallel()

    def get_dataset_parallel(
        self,
    ):
        # with concurrent.futures.ProcessPoolExecutor(
        #     max_workers=5,
        # ) as executor:
        #     datasets = tqdm(
        #         executor.map(
        #             self.get_dataset,
        #             self.dataset_configs,
        #             chunksize=1,
        #         )
        #     )
        #     datasets = list(datasets)
        #     print(datasets)
        #     for dataset in datasets:
        #         train, valid = dataset
        #         self.concat_dataset_train.extend(train)
        #         self.concat_dataset_test.extend(valid)
        datasets = Parallel(
            n_jobs=30,
            batch_size=1,
        )
        datasets = datasets(
            [delayed(self.get_dataset)(config) for config in self.dataset_configs]
        )
        for dataset in datasets:
            train, valid = dataset
            self.concat_dataset_train.extend(train)
            self.concat_dataset_test.extend(valid)

    def get_dataset(self, dataset_config):
        dataset_name = dataset_config["name"]
        status = dataset_config.get("status", "all")
        print(f"{dataset_name} - {status}")

        dataset = load_dataset(
            dataset_name,
            download_mode="force_redownload",
            keep_in_memory=True,
        )
        dataset = dataset["train"].filter(
            lambda item: self.filter_dataset(
                dataset_name=dataset_name,
                item=item,
                status=status,
            ),
            keep_in_memory=True,
            load_from_cache_file=False,
        )

        test_size = dataset_config["test_size"]
        dataset = dataset.train_test_split(
            test_size=test_size,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

        dataset_train = dataset["train"].to_list()
        dataset_test = dataset["test"].to_list()

        print("standart_dataset dataset_train...")
        dataset_train = self.standart_dataset(
            dataset=dataset_train,
            dataset_name=dataset_name,
        )

        print("standart_dataset dataset_test...")
        dataset_test = self.standart_dataset(
            dataset=dataset_test,
            dataset_name=dataset_name,
        )

        dataset_train = ChatDatasetVerbalist(
            original_records=dataset_train,
            tokenizer=self.tokenizer,
            templates_path=self.templates_path,
            test_size=test_size,
            max_tokens_count=self.max_tokens_count,
            dataset_type="train",
            status=status,
        )
        dataset_test = ChatDatasetVerbalist(
            original_records=dataset_test,
            tokenizer=self.tokenizer,
            templates_path=self.templates_path,
            max_tokens_count=self.max_tokens_count,
            test_size=test_size,
            dataset_type="test",
            status=status,
        )

        # self.concat_dataset_train.extend(dataset_train.records)
        # self.concat_dataset_test.extend(dataset_test.records)
        print(
            f"{dataset_name} -> train={len(dataset_train.records)} valid={len(dataset_test.records)}"
        )
        print("=" * 100)
        print("=" * 100)
        print("=" * 100)
        return [
            dataset_train.records,
            dataset_test.records,
        ]

    def get_datasets(self):
        for dataset_config in self.dataset_configs:
            dataset_name = dataset_config["name"]
            status = dataset_config.get("status", "all")
            print(f"{dataset_name} - {status}")

            dataset = load_dataset(
                dataset_name,
                download_mode="force_redownload",
                keep_in_memory=True,
            )
            dataset = dataset["train"].filter(
                lambda item: self.filter_dataset(
                    dataset_name=dataset_name,
                    item=item,
                    status=status,
                ),
                keep_in_memory=True,
                load_from_cache_file=False,
            )

            test_size = dataset_config["test_size"]
            dataset = dataset.train_test_split(
                test_size=test_size,
                keep_in_memory=True,
                load_from_cache_file=False,
            )

            dataset_train = dataset["train"].to_list()
            dataset_test = dataset["test"].to_list()

            print("standart_dataset dataset_train...")
            dataset_train = self.standart_dataset(
                dataset=dataset_train,
                dataset_name=dataset_name,
            )

            print("standart_dataset dataset_test...")
            dataset_test = self.standart_dataset(
                dataset=dataset_test,
                dataset_name=dataset_name,
            )

            dataset_train = ChatDatasetVerbalist(
                original_records=dataset_train,
                tokenizer=self.tokenizer,
                templates_path=self.templates_path,
                test_size=test_size,
                max_tokens_count=self.max_tokens_count,
                dataset_type="train",
                status=status,
            )
            dataset_test = ChatDatasetVerbalist(
                original_records=dataset_test,
                tokenizer=self.tokenizer,
                templates_path=self.templates_path,
                max_tokens_count=self.max_tokens_count,
                test_size=test_size,
                dataset_type="test",
                status=status,
            )

            self.concat_dataset_train.extend(dataset_train.records)
            self.concat_dataset_test.extend(dataset_test.records)
            print(
                f"{dataset_name} -> train={len(dataset_train.records)} valid={len(dataset_test.records)}"
            )
            print("=" * 100)
            print("=" * 100)
            print("=" * 100)

    def filter_dataset(
        self,
        dataset_name: str,
        item: dict,
        status: str,
    ):
        if status == "all":
            return True
        elif status == "ok":
            filter_dict = {
                "dim/oasst_en": self.default_filter,
                "dim/oasst_ru": self.default_filter,
                "dim/logic_tasks_ru": self.default_filter,
                "dim/logic_tasks_ru": lambda item: item["ok/trash"] == "ok",
            }
            filter_func = filter_dict[dataset_name]

            return filter_func(
                item=item,
            )
        else:
            assert False, "Invalid status"

    def default_filter(self, item):
        return item["status"] == "ok"

    def __getitem__(self, pos):
        return self.concat_dataset_train[pos]

    def __len__(self):
        return len(self.concat_dataset_train)

    def standart_dataset(
        self,
        dataset=None,
        dataset_name=None,
    ):
        convertsion_functions = {
            "dim/oasst_en": self.default_convert,
            "dim/oasst_ru": self.default_convert,
            "dim/lima": self.lima,
            "dim/logic_tasks_ru": self.logic_tasks_ru,
            "dim/what_where_when_3k": self.what_where_when_3k,
            "dim/competition_math_selected": self.competition_math_selected,
            "dim/wikihow_en": self.wikihow,
            "dim/wikihow_ru": self.wikihow,
            "dim/leetcodesolutions_en_2k": self.leetcodesolutions_en_2k,
            "dim/sharegpt_short_en": self.sharegpt_short,
            "dim/sharegpt_short_ru": self.sharegpt_short,
            "dim/roleplay_instruct_v2_final": self.roleplay_instruct_v2_final,
            "dim/ru_turbo_alpaca_evol_instruct_3k": self.ru_turbo_alpaca_evol_instruct_3k,
            "dim/ru_turbo_saiga_3k": self.ru_turbo_saiga_3k,
            "dim/essayforum_writing_prompts_6k": self.essayforum_writing_prompts_6k,
            "dim/openreview_prompts_65": self.openreview_prompts_65,
            "dim/kinomania_scripts": self.kinomania_scripts,
            "dim/bugurt_thread_prompts": self.bugurt_thread_prompts,
            "dim/russian_lyrics_prompts": self.russian_lyrics_prompts,
            "dim/ru_instruct_gpt4": self.ru_instruct_gpt4,
            "dim/gpt_roleplay_realm": self.gpt_roleplay_realm,
            "dim/ultrachat_ru": self.ultrachat_ru,
            "dim/tldr_17_3k": self.tldr_17_3k,
            "dim/grade_school_math_instructions_3k": self.grade_school_math_instructions_3k,
            "dim/tldr_news_3k": self.tldr_news_3k,
            "dim/scitldr": self.scitldr,
            "dim/linux_man_pages_tldr_summarized": self.linux_man_pages_tldr_summarized,
            "dim/grade_school_math_instructions_ru_3k": self.grade_school_math_instructions_ru_3k,
            "dim/dialogsum_ru_3k": self.dialogsum_ru_3k,
            "dim/dialogsum_3k": self.dialogsum_3k,
            "dim/dolphin_ru_3k": self.dolphin_ru_3k,
            "dim/dolphin_flan1m_alpaca_uncensored_3k": self.dolphin_flan1m_alpaca_uncensored_3k,
            "dim/HC3_ru_8k": self.HC3_ru_8k,
            "dim/ru_word_games_3k": self.ru_word_games_3k,
            "dim/runne_prompts": self.runne_prompts,
            "dim/horoscopes_ru_1k": self.horoscopes_ru_1k,
            "dim/huggingartists_prompts": self.huggingartists_prompts,
            "dim/lurk_prompts": self.lurk_prompts,
            "dim/yandex_q_10k": self.yandex_q_10k,
            "dim/panorama_prompts": self.panorama_prompts,
            "dim/resh_edu_short_prompts": self.resh_edu_short_prompts,
            "dim/bugurt_completion_prompts": self.bugurt_completion_prompts,
        }

        return convertsion_functions[dataset_name](dataset)

    def default_convert(self, dataset):
        return dataset

    def lima(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = dataset[i].pop("conversations")

        return dataset

    def logic_tasks_ru(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []
            dataset[i][self.conversation_field].append(dataset[i]["task"])
            dataset[i][self.conversation_field].append(dataset[i]["answer"])

        return dataset

    def what_where_when_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = dataset[i]["question"]
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['explanation']}\n{dataset[i]['answer']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def competition_math_selected(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = dataset[i]["problem"]
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['solution']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def wikihow(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = dataset[i]["INSTRUCTION"]
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['RESPONSE']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def leetcodesolutions_en_2k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = dataset[i]["input"]
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['output']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def sharegpt_short(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = dataset[i]["conversation"]

        return dataset

    def roleplay_instruct_v2_final(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['instruction']}\n{dataset[i]['input']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["output"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def ru_turbo_alpaca_evol_instruct_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = dataset[i]["instruction"].replace("Вход:", "")
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['output']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def ru_turbo_saiga_3k(self, dataset):
        new_dataset = []
        for i in range(len(dataset)):
            content = dataset[i]["messages"]["content"]
            if len(content) > 0:
                new_dataset.append(
                    {
                        self.conversation_field: content,
                    }
                )

        return new_dataset

    def essayforum_writing_prompts_6k(self, dataset):
        selected_prompts = [
            "rate the essay based on its clarity, organization, and use of evidence to support the argument. Justify your rating with specific examples from the essay and evaluation.",
            "Rate the essay based on the clarity of the writer's opinion, development of reasoning paragraphs, and effectiveness of the concluding summary.",
            "Rate the essay based on the writer's ability to present a clear and cohesive argument, provide relevant examples and reasoning, and effectively address the topic.",
            "Evaluate the essay provided in terms of coherence, cohesiveness, and the appropriateness of the response to the given question.",
            "Evaluate the essay by discussing its strengths and weaknesses.",
            "Evaluate the essay provided above. Assess the writer's ability to follow the given instructions and provide a well-structured and coherent response.",
            "Evaluate the essay based on the given prompt and provide your analysis.",
            "Please rate the essay based on how clear it is, how well it is organized, and how effectively it uses evidence to support the argument. Please justify your rating with specific examples from the essay and your evaluation.",
            "Hey there! Can you please rate this essay for me? I'd like you to consider how clear it is, how well it's organized, and how effectively it uses evidence to support its argument. It would be great if you could back up your rating with specific examples from the essay and explain your evaluation. Thanks a bunch!",
            "Rate the essay based on how clear the writer's opinion is, how well they develop their reasoning paragraphs, and how effective their concluding summary is.",
            "Rate the essay based on the writer's ability to present a clear and cohesive argument, provide relevant examples and reasoning, and effectively address the topic. Can you do that for me, please?",
            "Hey there! Can you please take a look at this essay and tell me what you think about its coherence, cohesiveness, and how well it responds to the given question? Thanks!",
            "Hey there! Can you take a look at this essay and give me your thoughts on it? I'd love to hear about its strengths and weaknesses.",
            "Can you please evaluate the essay above? I want to know if the writer followed the instructions properly and if their response is well-structured and coherent.",
            "hey, can you read this essay and tell me what you think?",
            "Rate essay on clarity, organization, and evidence.",
            "Rate essay on clarity, development, and conclusion.",
            "Rate essay on clarity, examples, reasoning, and topic.",
            "Evaluate coherence, cohesiveness, and appropriateness of essay.",
            "Evaluate essay's strengths and weaknesses.",
            "Evaluate essay structure and coherence.",
            "Evaluate and analyze an essay based on a prompt.",
            "Rating essay clarity, organization, and evidence usage.",
            "Rate essay on clarity, organization, and evidence.",
            "Rate essay on clarity, reasoning development, and conclusion.",
            "Rate essay on clarity, examples, reasoning, and addressing topic.",
            "Evaluate essay coherence, cohesiveness, and relevance.",
            "Evaluate essay's strengths and weaknesses.",
            "Evaluate essay structure and coherence.",
            "Request for feedback on an essay.",
            "evaluate this essay",
            "evaluate essay",
            "rate essay",
        ]

        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []
            generated_instruction = selected_prompts[i % len(selected_prompts)]

            question = dataset[i]["prompt"]
            question = f"{question}\n\n{generated_instruction}"
            dataset[i][self.conversation_field].append(question)

            answer = f"{dataset[i]['answer']}"
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def openreview_prompts_65(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['latex']}\n\n{dataset[i]['help_prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["full_review"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def kinomania_scripts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}\n\n{dataset[i]['movie_description']}"
            # question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["movie_script"][:6000]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def bugurt_thread_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            # question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["bugurt"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def russian_lyrics_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["solution"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def ru_instruct_gpt4(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["solution"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def gpt_roleplay_realm(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = dataset[i]["conversation"]

        return dataset

    def ultrachat_ru(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = dataset[i]["conversation"]

        return dataset

    def tldr_17_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['content']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["summary"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def grade_school_math_instructions_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['INSTRUCTION']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["RESPONSE"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def tldr_news_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['content']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["headline"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def scitldr(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['source']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["target"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def linux_man_pages_tldr_summarized(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['Text']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["Summary"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def grade_school_math_instructions_ru_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['question']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["answer"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def dialogsum_ru_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['dialogue']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["summary"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def dialogsum_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['dialogue']} TL;DR"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["summary"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def dolphin_ru_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['input']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["output"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def dolphin_flan1m_alpaca_uncensored_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['input']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["output"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def HC3_ru_8k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['question']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["human_answers"][0]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def ru_word_games_3k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["answer"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def runne_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = (
                f"Найди все именованные сущности в данном тексте: {dataset[i]['text']}"
            )
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["parsed_entities"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def horoscopes_ru_1k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["prediction"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def huggingartists_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["song"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def lurk_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["text"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def yandex_q_10k(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['description']}\n{dataset[i]['question']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["answer"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def panorama_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["text"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def resh_edu_short_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["solution"]
            dataset[i][self.conversation_field].append(answer)

        return dataset

    def bugurt_completion_prompts(self, dataset):
        for i in range(len(dataset)):
            dataset[i][self.conversation_field] = []

            question = f"{dataset[i]['prompt']}"
            dataset[i][self.conversation_field].append(question)

            answer = dataset[i]["bugurt"]
            dataset[i][self.conversation_field].append(answer)

        return dataset
