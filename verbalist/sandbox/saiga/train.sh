export CUDA_VISIBLE_DEVICES=3
export WANDB_BASE_URL="https://api.wandb.ai"

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
	echo "Please input model folder name"
    exit
fi

mkdir -p ./models/
mkdir -p ./models/temp

mkdir -p ./models/$OUTPUT
export WANDB_NAME=$OUTPUT

nohup python -u -m src.train --config-file configs/saiga2_7b.json \
	--train-file ./train.jsonl \
	--val-file valid.jsonl  \
	--output-dir models/$OUTPUT \
	--omit-base-model-save > ./models/$OUTPUT/training.log &