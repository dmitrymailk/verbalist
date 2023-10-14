# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=3


# python -u mmlu_ru.py --hf_model_id "huggyllama/llama-7b" --k_shot 5 --lang "ru" --output_dir "results"
model_name=/home/kosenko/verbalist/verbalist/model/models/verbalist_7b_v7/checkpoint-16500/adapter_model
# model_name=Open-Orca/Mistral-7B-OpenOrca
# lang="en"
lang="ru"
results_folder="verbalist_7b_v7_checkpoint-16500_flash_$lang"
# results_folder="open_orca_mistral_7b_$lang"
# python -u mmlu_ru.py --hf_model_id $model_name --k_shot 5 --lang "ru" --output_dir $results_folder
python -u -m mmlu_ru --hf_model_id $model_name --k_shot 5 --lang $lang --output_dir $results_folder
