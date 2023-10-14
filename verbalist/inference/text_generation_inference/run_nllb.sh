model=facebook/nllb-200-3.3B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus 1 \
 	--shm-size 1g \
	-p 8081:80 \
	-v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.0 \
	--model-id $model