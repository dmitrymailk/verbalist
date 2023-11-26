import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import gc


class VerbalistConversation:
    def __init__(
        self,
        message_template="<s> {role}\n{content} </s> \n",
        system_prompt="Ты — Буквоед, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=9225,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])

        return final_text.strip()


class VerbalistOpenchatConversation:
    def __init__(
        self,
        message_template="<s> {role}\n{content} </s> \n",
        system_prompt="Ты — Буквоед, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=12435,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system ", "content": system_prompt}]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({"role": "user high_quality", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot high_quality", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += "<s> bot high_quality"

        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()
