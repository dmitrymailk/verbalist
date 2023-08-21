import argparse
import random
import json
import os

import wandb
import torch
import numpy as np
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
)
from transformers import (
    Trainer,
    TrainingArguments,
    logging,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from src.dataset import ChatDatasetSaiga, ChatDatasetVerbalistUnion
from src.util.dl import set_random_seed, fix_tokenizer, fix_model
from src.util.io import read_jsonl

from src.flash import patch_model

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainerNoBaseSave(Trainer):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        print("Running custom _save_checkpoint")
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control


def custom_prepare_model_for_int8_training(
    model, output_embedding_layer_name="lm_head", layer_norm_names=["layer_norm"]
):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(
            layer_norm_name in name for layer_norm_name in layer_norm_names
        ):
            param.data = param.data.to(torch.float32)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(
            model,
            output_embedding_layer_name,
            CastOutputToFloat(output_embedding_layer),
        )

    model.gradient_checkpointing_enable()

    return model


def train(
    config_file,
    checkpoint,
    train_file,
    val_file,
    train_sample_rate,
    val_sample_rate,
    output_dir,
    report_to,
    seed,
    local_rank,
    omit_base_model_save,
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_file, "r") as r:
        config = json.load(r)

    device_map = "auto"

    deepspeed_config = config.get("deepspeed")
    trainer_config = config["trainer"]
    lora_config = config.get("lora")
    callbacks = [SavePeftModelCallback] if lora_config else []
    training_args = TrainingArguments(
        output_dir=output_dir,
        # save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to,
        ddp_find_unused_parameters=None,
        deepspeed=deepspeed_config,
        **trainer_config,
    )
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer = fix_tokenizer(tokenizer)
    tokenizer.save_pretrained(output_dir)

    model_type = config.get("model_type", "causal")
    templates_path = config.get("templates_path", "saiga2_7b.json")
    only_target_loss = config.get("only_target_loss", True)
    mode = config.get("mode", "saiga_chat")
    use_flash = config.get("use_flash", False)
    max_tokens_count = config["max_tokens_count"]

    if mode == "saiga_chat":
        train_records = read_jsonl(train_file)
        val_records = read_jsonl(val_file)
        random.shuffle(train_records)
        print(train_records[0])

        train_dataset = ChatDatasetSaiga(
            train_records,
            tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=train_sample_rate,
            templates_path=templates_path,
            only_target_loss=only_target_loss,
            dataset_type="train",
        )

        val_dataset = ChatDatasetSaiga(
            val_records,
            tokenizer,
            max_tokens_count=max_tokens_count,
            sample_rate=train_sample_rate,
            templates_path=templates_path,
            only_target_loss=only_target_loss,
            dataset_type="valid",
        )
    elif mode == "verbalist_chat":
        # datasets_configs = config["datasets_configs"]
        datasets_configs = [
            # {
            #     "name": "dim/oasst_en",
            #     "status": "ok",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/oasst_ru",
            #     "status": "ok",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/lima",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/logic_tasks_ru",
            #     "status": "ok",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/what_where_when_3k",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/competition_math_selected",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/wikihow_en",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/wikihow_ru",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/leetcodesolutions_en_2k",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/sharegpt_short_en",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/sharegpt_short_ru",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/roleplay_instruct_v2_final",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/ru_turbo_alpaca_evol_instruct_3k",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/ru_turbo_saiga_3k",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/essayforum_writing_prompts_6k",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/openreview_prompts_65",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/openreview_prompts_65",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/kinomania_scripts",
            #     "status": "all",
            #     "test_size": 1,
            # },
            # {
            #     "name": "dim/bugurt_thread_prompts",
            #     "status": "all",
            #     "test_size": 1,
            # },
            {
                "name": "dim/russian_lyrics_prompts",
                "status": "all",
                "test_size": 1,
            },
        ]

        union_dataset = ChatDatasetVerbalistUnion(
            dataset_configs=datasets_configs,
            tokenizer=tokenizer,
            templates_path=templates_path,
            max_tokens_count=max_tokens_count,
        )

        train_dataset = union_dataset.concat_dataset_train
        val_dataset = union_dataset.concat_dataset_test
        print(f"Train length={len(train_dataset)} Valid length={len(val_dataset)}")
    else:
        assert False

    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        pad_to_multiple_of=8,
    )

    print("INPUT_IDS")
    print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
    print("MASK")
    print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
    print("LABELS")
    print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

    model_types = {"causal": AutoModelForCausalLM, "seq2seq": AutoModelForSeq2SeqLM}
    load_in_8bit = bool(config.get("load_in_8bit", False))
    load_in_4bit = bool(config.get("load_in_4bit", False))
    use_bf16 = bool(trainer_config.get("bf16", False))
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if load_in_8bit:
        assert not load_in_4bit
        model = model_types[model_type].from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        model = fix_model(model, tokenizer, use_resize=False)
        model = custom_prepare_model_for_int8_training(model)
        # model = prepare_model_for_kbit_training(model)

    elif load_in_4bit:
        assert not load_in_8bit
        model = model_types[model_type].from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch_dtype,
        )
        model = fix_model(model, tokenizer, use_resize=False)
        # model = prepare_model_for_kbit_training(model)
        model = custom_prepare_model_for_int8_training(model)
    else:
        model = model_types[model_type].from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        model = fix_model(model, tokenizer)

    if use_flash:
        patch_model(model)
    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_tokens_count

    model.is_parallelizable = True
    model.model_parallel = True

    if lora_config:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)

    trainer_class = Trainer if not omit_base_model_save else TrainerNoBaseSave
    print("Trainer class:", trainer_class)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        data_collator=data_collator,
    )

    with wandb.init(project="verbalist", name=config_file) as run:
        trainer.train(checkpoint)
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--val-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--omit-base-model-save", action="store_true", default=False)
    args = parser.parse_args()
    train(**vars(args))
