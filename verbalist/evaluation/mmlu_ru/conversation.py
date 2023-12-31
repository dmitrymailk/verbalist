import abc
import typing as tp


class Conversation(abc.ABC):
    """
    Inspired by https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
    """

    def __init__(self, system_prompt: str, roles: tp.Tuple[str, str]):
        self.system_prompt = system_prompt
        self.roles = roles
        self.messages: tp.List[tp.Tuple[str, str]] = []

    @abc.abstractmethod
    def get_prompt(self) -> str:
        pass

    def update_last_message(self, text: str) -> None:
        self.messages[-1] = (self.messages[-1][0], text)

    def append_message(self, role: str, text: str) -> None:
        self.messages.append([role, text])


class LlamaConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="",  # faked for compatibility
            roles=("", ""),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        for role, text in self.messages:
            if text:
                prompt += f" {role} {text}"
            else:
                prompt += f" {role} "
        return prompt


class SaigaConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="<s> system\n Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.</s>\n",
            roles=("user", "bot"),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        for role, text in self.messages:
            if text:
                prompt += f"<s> {role}\n {text} </s>\n"
            else:
                prompt += f"<s> {role} "
        return prompt


class VerbalistConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="<s> system\n Ты — Буквоед, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. </s>\n",
            roles=("user", "bot"),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        for role, text in self.messages:
            if text:
                prompt += f"<s> {role}\n {text} </s>\n"
            else:
                prompt += f"<s> {role} "
        return prompt


class MistralOrcaConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="<|im_start|> system\n You are MistralOrca, a large language model trained by Alignment Lab AI. <|im_end|>\n",
            roles=("user", "assistant"),
        )

    def get_prompt(self) -> str:
        # prompt = self.system_prompt
        prompt = ""
        for role, text in self.messages:
            if text:
                prompt += f"<|im_start|>{role}\n{text}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>{role}\n"
        return prompt


class TinyllamaOrcaConversation(Conversation):
    def __init__(self):
        super().__init__(
            system_prompt="<s>[INST] <<SYS>>\n Вы помощник ИИ, который помогает людям находить информацию. <</SYS>>\n",
            roles=("user", "assistant"),
        )

    def get_prompt(self) -> str:
        prompt = self.system_prompt
        # prompt = ""
        for role, text in self.messages:
            if text:
                prompt += f"{text} [/INST]\n"
            else:
                prompt += f"\n"
        return prompt


conversation_classes = {
    "saiga": SaigaConversation,
    "verbalist": VerbalistConversation,
    "llama": LlamaConversation,
    "open_orca_mistral": MistralOrcaConversation,
    "tinyllama_open_orca": TinyllamaOrcaConversation,
}


if __name__ == "__main__":
    utterances = [
        "А тот второй парень — он тоже терминатор?",
        "Не совсем. Модель T-1000, усовершенствованный прототип.",
        "То есть, он более современная модель?",
        "Да. Мимикрирующий жидкий металл.",
        "И что это значит?",
    ]
    print("-=-=-=- Saiga -=-=-=-")
    conv = SaigaConversation()
    conv.append_message(conv.roles[0], utterances[0])
    conv.append_message(conv.roles[1], utterances[1])
    conv.append_message(conv.roles[0], utterances[2])
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())
    print("-=-=-=-=-=-=-=-=-=-=-")
