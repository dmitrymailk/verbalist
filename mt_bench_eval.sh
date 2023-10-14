export CUDA_VISIBLE_DEVICES=1

log_path="verbalist/evaluation/mt_bench/logs/eval$(date +"%d.%m.%Y_%H:%M:%S").log"

# nohup python -u verbalist/evaluation/mt_bench/mt_bench_evaluation.py > $log_path &
nohup python -u -m verbalist.evaluation.mt_bench.mt_bench_evaluation > $log_path &
# python -u -m verbalist.evaluation.mt_bench.mt_bench_evaluation
