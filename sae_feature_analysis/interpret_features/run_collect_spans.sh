MODEL="llama"
TTL_GROUP=4
SAE_PATH="xxx/outputs/xxx.pth"
DATA_PATH="xxx.tsv"
# 0.0 0.5 1.0 1.5 2.0 4.0
THRESHOLD_LIST=(0.0)

for THRESHOLD in "${THRESHOLD_LIST[@]}"; do
  echo "Running threshold = ${THRESHOLD}"

  CUDA_VISIBLE_DEVICES=0 nohup python collect_sapns.py 0 ${MODEL} 0 ${TTL_GROUP} \
    --data-path ${DATA_PATH} --threshold ${THRESHOLD} \
    --sae-path ${SAE_PATH} > RQ4_Logs/logs_${MODEL}_group0_thr${THRESHOLD}.out 2>&1 &
  
  CUDA_VISIBLE_DEVICES=1 nohup python collect_spans.py 0 ${MODEL} 1 ${TTL_GROUP} \
    --data-path ${DATA_PATH} --threshold ${THRESHOLD} \
    --sae-path ${SAE_PATH} > RQ4_Logs/logs_${MODEL}_group1_thr${THRESHOLD}.out 2>&1 &

  CUDA_VISIBLE_DEVICES=2 nohup python collect_spans.py 0 ${MODEL} 2 ${TTL_GROUP} \
    --data-path ${DATA_PATH} --threshold ${THRESHOLD} \
    --sae-path ${SAE_PATH} > RQ4_Logs/logs_${MODEL}_group2_thr${THRESHOLD}.out 2>&1 &

  CUDA_VISIBLE_DEVICES=3 nohup python collect_spans.py 0 ${MODEL} 3 ${TTL_GROUP} \
    --data-path ${DATA_PATH} --threshold ${THRESHOLD} \
    --sae-path ${SAE_PATH} > RQ4_Logs/logs_${MODEL}_group3_thr${THRESHOLD}.out 2>&1 &

  tail -f RQ4_Logs/logs_${MODEL}_group0_thr${THRESHOLD}.out

done

echo "All runs finished!"

