#!/bin/bash
set -euo pipefail

# ============================================================
# Global config
# ============================================================
PYTHON=python

TRAIN_SCRIPT=g_ani_train_rampup_ema_basis.py
SAMPLE_SCRIPT_EDM=g_sample_path_trainednet_ani_edmsampler_v2_basis.py
SAMPLE_SCRIPT_STD=g_sample_path_basis.py
FID_PY=fid.py

CKPT_ROOT=finetune_initial
OUTDIR=samples_rebuttal
LOGDIR=logs_rebuttal
mkdir -p "${LOGDIR}"

# Training / sampling config
RAMPUP=10000
KIMG=1200
GRAD_ACCUM=1

BATCH=2048
START_SEED=0
NUM_SEEDS=50000
STEPS_LIST=(5 6 7 8 9 18 30 40)

# ============================================================
# Datasets to run
# ============================================================
DATASETS=(
  cifar10
)

# ============================================================
# Main loop
# ============================================================
for DATASET in "${DATASETS[@]}"; do

  MODEL_TAG="g-ani-rampup-${RAMPUP}-ema-basis"
  LOGFILE="${LOGDIR}/log_${DATASET}_${MODEL_TAG}.txt"

  {
    echo "========================================"
    echo "Dataset      : ${DATASET}"
    echo "Model tag    : ${MODEL_TAG}"
    echo "Checkpoint   : ${CKPT_ROOT}"
    echo "Output dir   : ${OUTDIR}"
    echo "Log file     : ${LOGFILE}"
    echo "========================================"
  } | tee "${LOGFILE}"

  # ----------------------------------------------------------
  # Select FID reference
  # ----------------------------------------------------------
  case "${DATASET}" in
    cifar10)
      FID_REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
      ;;
    afhqv2)
      FID_REF="afhq/fid-refs/afhqv2-64x64.npz"
      ;;
    ffhq)
      FID_REF="ffhq/fid-refs/ffhq-64x64.npz"
      ;;
    *)
      echo "[ERROR] Unknown dataset: ${DATASET}" | tee -a "${LOGFILE}"
      exit 1
      ;;
  esac

  echo "[INFO] FID reference: ${FID_REF}" | tee -a "${LOGFILE}"

  # ----------------------------------------------------------
  # 1) Training
  # ----------------------------------------------------------
  echo "[TRAIN] Start training on ${DATASET}" | tee -a "${LOGFILE}"

  "${PYTHON}" "${TRAIN_SCRIPT}" \
    --dataset "${DATASET}" \
    --out "${CKPT_ROOT}" \
    --kimg "${KIMG}" \
    --lr_rampup_kimg "${RAMPUP}" \
    --per_label_pca \
    --grad_accum "${GRAD_ACCUM}" \
    2>&1 | tee -a "${LOGFILE}"

  # ----------------------------------------------------------
  # 2) Sampling + FID (EDM sampler)
  # ----------------------------------------------------------
  echo "[INFO] Sampling with EDM sampler" | tee -a "${LOGFILE}"

  for STEPS in "${STEPS_LIST[@]}"; do
    {
      echo "----------------------------------------"
      echo "[EDM SAMPLE] STEPS = ${STEPS}"
      echo "----------------------------------------"
    } | tee -a "${LOGFILE}"

    "${PYTHON}" "${SAMPLE_SCRIPT_EDM}" \
      --dataset "${DATASET}" \
      --model_tag "${MODEL_TAG}" \
      --ckpt_root "${CKPT_ROOT}" \
      --out "${OUTDIR}" \
      --batch "${BATCH}" \
      --start_seed "${START_SEED}" \
      --num_seeds "${NUM_SEEDS}" \
      --steps "${STEPS}" \
      2>&1 | tee -a "${LOGFILE}"

    SAMPLE_DIR="${OUTDIR}/${DATASET}-${MODEL_TAG}-trained-ani-edmsampler-v2-seed-${START_SEED}/fid-heun-${MODEL_TAG}-trained-ani-edmsampler-v2-${STEPS}"

    echo "[FID][EDM] STEPS=${STEPS}" | tee -a "${LOGFILE}"

    set +e
    torchrun --standalone --nproc_per_node=1 \
      "${FID_PY}" calc \
      --images="${SAMPLE_DIR}" \
      --ref="${FID_REF}" \
      2>&1 | tee -a "${LOGFILE}"
    FID_STATUS=$?
    set -e

    if [[ ${FID_STATUS} -ne 0 ]]; then
      echo "[WARNING] FID failed (EDM) | STEPS=${STEPS}" | tee -a "${LOGFILE}"
    fi
  done

  # ----------------------------------------------------------
  # 3) Sampling + FID (standard sampler)
  # ----------------------------------------------------------
  echo "[INFO] Sampling with standard sampler" | tee -a "${LOGFILE}"

  for STEPS in "${STEPS_LIST[@]}"; do
    {
      echo "----------------------------------------"
      echo "[STD SAMPLE] STEPS = ${STEPS}"
      echo "----------------------------------------"
    } | tee -a "${LOGFILE}"

    "${PYTHON}" "${SAMPLE_SCRIPT_STD}" \
      --dataset "${DATASET}" \
      --model_tag "${MODEL_TAG}" \
      --ckpt_root "${CKPT_ROOT}" \
      --out "${OUTDIR}" \
      --batch "${BATCH}" \
      --start_seed "${START_SEED}" \
      --num_seeds "${NUM_SEEDS}" \
      --steps "${STEPS}" \
      2>&1 | tee -a "${LOGFILE}"

    SAMPLE_DIR="${OUTDIR}/${DATASET}-${MODEL_TAG}/fid-heun-${MODEL_TAG}-${STEPS}"

    echo "[FID][STD] STEPS=${STEPS}" | tee -a "${LOGFILE}"

    set +e
    torchrun --standalone --nproc_per_node=1 \
      "${FID_PY}" calc \
      --images="${SAMPLE_DIR}" \
      --ref="${FID_REF}" \
      2>&1 | tee -a "${LOGFILE}"
    FID_STATUS=$?
    set -e

    if [[ ${FID_STATUS} -ne 0 ]]; then
      echo "[WARNING] FID failed (STD) | STEPS=${STEPS}" | tee -a "${LOGFILE}"
    fi
  done


  {
    echo "========================================"
    echo "Finished dataset: ${DATASET}"
    echo "========================================"
  } | tee -a "${LOGFILE}"

done

echo "ALL DATASETS DONE."
