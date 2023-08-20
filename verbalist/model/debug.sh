export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=2
export WANDB_BASE_URL="https://api.wandb.ai"

OUTPUT=debug_model

mkdir -p ./models/
mkdir -p ./models/temp

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

python -u -m src.train --config-file configs/verbalist_7b.json \
	--train-file ./train.jsonl \
	--val-file valid.jsonl  \
	--output-dir models/$OUTPUT \
	--omit-base-model-save 
	
	# > ./models/$OUTPUT/training.log 