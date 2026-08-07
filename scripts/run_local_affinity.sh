python affinity/predict_affinity.py \
  --affinity-input runs/full_8v1q/10_affinity_input \
  --output-csv runs/full_8v1q/10_affinity_input/model_pose_predictions.csv \
  --summary-json runs/full_8v1q/10_affinity_input/local_affinity_summary.json \
  --install-filled-template \
  --resume \
  --device "${AFFINITY_DEVICE:-cuda:0}" \
  --batch-size "${AFFINITY_BATCH_SIZE:-1}" \
  --threads "${AFFINITY_THREADS:-0}" \
  --progress-every "${AFFINITY_PROGRESS_EVERY:-100}"
