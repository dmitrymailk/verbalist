# export CUDA_VISIBLE_DEVICES="1,2,3"
# export CUDA_VISIBLE_DEVICES="2,3"
export CUDA_VISIBLE_DEVICES="1"

# nohup python -u slim_orca.py > ./logs/translation.log &
nohup python -u slim_orca.py > ./logs/translation_2.log &
# nohup python -u slim_orca.py > ./logs/translation_3.log &