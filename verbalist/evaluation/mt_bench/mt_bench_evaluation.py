import json

from datasets import load_dataset
from tqdm import tqdm

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

from verbalist.generation.generation_utils import VerbalistConversation, generate

from datasets import load_dataset


def evaluate_mt_bench_ru(
    weights_path,
    tokenizer_path,
    output_save_path,
):
    dataset = load_dataset("dim/mt_bench_ru")
    dataset = dataset["train"]
    dataset = dataset.to_list()
    config = PeftConfig.from_pretrained(weights_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_8bit=True,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        weights_path,
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    generation_config = GenerationConfig(
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=2048,
        no_repeat_ngram_size=20,
        repetition_penalty=1.1,
        temperature=0.5,
        top_k=30,
        top_p=0.9,
        # do_sample=True,
        num_beams=1
    )
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        conversation = VerbalistConversation()
        dataset[i]["replies"] = []
        for turn in item["turns_ru"]:
            conversation.add_user_message(turn)
            prompt = conversation.get_prompt(tokenizer)
            # print("PROMPT", prompt)
            print("USER: ", turn)
            # print("USER: ", prompt)
            output = generate(
                model,
                tokenizer,
                prompt,
                generation_config,
            )
            # print(inp)
            print("BOT: ", output)
            conversation.add_bot_message(output)
            dataset[i]["replies"].append(output)
            print()
            print("==============================")
            print()
    with open(
        output_save_path,
        "w",
        encoding="utf-8",
    ) as outfile:
        json.dump(dataset, outfile)


if __name__ == "__main__":
    weights_path = (
        "verbalist/model/models/verbalist_7b_v7/checkpoint-25000/adapter_model"
    )
    model_name = "verbalist_7b_v7_checkpoint_25000.json"
    output_save_path = f"verbalist/evaluation/mt_bench/llm_judge/data/mt_bench/model_answer/{model_name}"
    evaluate_mt_bench_ru(
        weights_path=weights_path,
        tokenizer_path=weights_path,
        output_save_path=output_save_path,
    )
