export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=2
# export CUDA_VISIBLE_DEVICES=3
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

# nohup python -u -m src.train --config-file configs/verbalist_65b.json \
# nohup python -u -m src.train --config-file configs/verbalist_1.3b.json \
# nohup python -u -m src.train --config-file configs/verbalist_30b.json \
# nohup python -u -m src.train --config-file configs/verbalist_13b.json \
# nohup python -u -m src.train --config-file configs/verbalist_7b.json \
# nohup python -u -m src.train --config-file configs/verbalist_70b.json \
# nohup python -u -m src.train --config-file configs/mistral_7b_oasst1_dolly.json \
# nohup python -u -m src.train --config-file configs/verbalist_1.1b.json \
# nohup python -u -m src.train --config-file configs/verbalist_7b.json \
nohup python -u -m src.train --config-file configs/verbalist_openchat_7b.json \
	--output-dir models/$OUTPUT \
	--omit-base-model-save > ./models/$OUTPUT/training.log &