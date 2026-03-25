#!/bin/bash
set -e

# ============================
# User config
# ============================
PYTHON=python
TRAIN_SCRIPT=g_iso_train_rampup_ema.py
SCRIPT=g_sample_path_trainednet_edmsampler.py

DATASET=cifar10
RAMPUP=10000
MODEL_TAG=g-iso-rampup-${RAMPUP}-ema
CKPT_ROOT=finetune_initial
OUTDIR=samples_rebuttal

BATCH=2048
START_SEED=0
NUM_SEEDS=50000

STEPS_LIST=(5 6 7 8 9 18 30 40)

# ============================
# FID config (NVLabs style)
# ============================
FID_PY=fid.py
FID_REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"

# ============================
# Run
# ============================
$PYTHON $TRAIN_SCRIPT \
  --dataset $DATASET \
  --out $CKPT_ROOT \
  --lr 1e-5 \
  --kimg 1200 \
  --grad_accum 4 

for STEPS in "${STEPS_LIST[@]}"; do
  echo "========================================"
  echo "Sampling with STEPS = $STEPS"
  echo "========================================"

  # 1) Sampling
  $PYTHON $SCRIPT \
    --dataset $DATASET \
    --model_tag $MODEL_TAG \
    --ckpt_root $CKPT_ROOT \
    --out $OUTDIR \
    --batch $BATCH \
    --start_seed $START_SEED \
    --num_seeds $NUM_SEEDS \
    --steps $STEPS

  SAMPLE_DIR=${OUTDIR}/${DATASET}-${MODEL_TAG}-trained-edmsampler-seed-${START_SEED}/fid-heun-${MODEL_TAG}-trained-edmsampler-${STEPS}

  # echo "Images saved to: ${SAMPLE_DIR}"

  # # 2) FID
  echo "Computing FID for STEPS = $STEPS"

  set +e
  torchrun --standalone --nproc_per_node=1 \
    ${FID_PY} calc \
    --images=${SAMPLE_DIR} \
    --ref=${FID_REF}
  FID_STATUS=$?
  set -e

  if [ $FID_STATUS -ne 0 ]; then
    echo "WARNING: FID computation failed for STEPS=${STEPS}"
  fi

done

echo "========================================"
echo "All done."
