#!/bin/bash
set -euo pipefail

# ============================================================
# Global config
# ============================================================
PYTHON=python

# -------- training scripts --------
TRAIN_ISO_SCRIPT=g_iso_train_rampup_ema_condition_debug.py
TRAIN_ANI_SCRIPT=g_ani_train_rampup_ema_condition.py

# -------- EDM sampling scripts (DIFFERENT!) --------
SAMPLE_EDM_ISO=g_sample_path_trainednet_edmsampler.py
SAMPLE_EDM_ANI=g_sample_path_trainednet_ani_edmsampler_v2_condition.py

# -------- standard sampler (shared) --------
SAMPLE_SCRIPT_STD=g_sample_path_condition.py

# -------- FID --------
FID_PY=fid.py

# -------- model tags --------
ISO_TAG="g-iso-condition-debug"
ANI_TAG="g-ani-condition"

# -------- dirs --------
CKPT_ROOT=finetune_initial
OUTDIR=samples_rebuttal
LOGDIR=logs_rebuttal
mkdir -p "${LOGDIR}"

# -------- training / sampling config --------
KIMG=1200
GRAD_ACCUM=1

BATCH=2048
START_SEED=0
NUM_SEEDS=50000
STEPS_LIST=(5 6 7 8 9 18 30 40)

# ============================================================
# Datasets
# ============================================================
DATASETS=(
  cifar10
)

# ============================================================
# Main loop
# ============================================================
for DATASET in "${DATASETS[@]}"; do

  LOGFILE="${LOGDIR}/log_${DATASET}_${ISO_TAG}_and_${ANI_TAG}.txt"

  {
    echo "========================================"
    echo "Dataset        : ${DATASET}"
    echo "ISO model tag  : ${ISO_TAG}"
    echo "ANI model tag  : ${ANI_TAG}"
    echo "Checkpoint dir : ${CKPT_ROOT}"
    echo "Output dir     : ${OUTDIR}"
    echo "Log file       : ${LOGFILE}"
    echo "========================================"
  } | tee "${LOGFILE}"

  # ----------------------------------------------------------
  # FID reference
  # ----------------------------------------------------------
  case "${DATASET}" in
    cifar10)
      FID_REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
      ;;
    *)
      echo "[ERROR] Unknown dataset: ${DATASET}"
      exit 1
      ;;
  esac

  echo "[INFO] FID reference: ${FID_REF}" | tee -a "${LOGFILE}"

  # ==========================================================
  # 1) ISO TRAINING
  # ==========================================================
  echo "[TRAIN][ISO]" | tee -a "${LOGFILE}"
  "${PYTHON}" "${TRAIN_ISO_SCRIPT}" \
    --dataset "${DATASET}" \
    --out "${CKPT_ROOT}" \
    --kimg "${KIMG}" \
    --grad_accum "${GRAD_ACCUM}" \
    2>&1 | tee -a "${LOGFILE}"

  # ==========================================================
  # 2) ANI TRAINING
  # ==========================================================
  echo "[TRAIN][ANI]" | tee -a "${LOGFILE}"
  "${PYTHON}" "${TRAIN_ANI_SCRIPT}" \
    --dataset "${DATASET}" \
    --out "${CKPT_ROOT}" \
    --kimg "${KIMG}" \
    --grad_accum "${GRAD_ACCUM}" \
    2>&1 | tee -a "${LOGFILE}"

  # ==========================================================
  # 3) SAMPLING + FID
  # ==========================================================
  for MODEL_TAG in "${ISO_TAG}" "${ANI_TAG}"; do

    echo "========================================" | tee -a "${LOGFILE}"
    echo "[SAMPLE] MODEL = ${MODEL_TAG}" | tee -a "${LOGFILE}"
    echo "========================================" | tee -a "${LOGFILE}"

    # ----------------------------------------------------------
    # EDM sampler + SAMPLE_DIR (CRITICAL FIX)
    # ----------------------------------------------------------
    if [[ "${MODEL_TAG}" == "${ISO_TAG}" ]]; then
      EDM_SAMPLER="${SAMPLE_EDM_ISO}"
      SAMPLE_ROOT="${OUTDIR}/${DATASET}-${MODEL_TAG}-trained-edmsampler-seed-${START_SEED}"
      SAMPLE_SUB_PREFIX="fid-heun-${MODEL_TAG}-trained-edmsampler"
    else
      EDM_SAMPLER="${SAMPLE_EDM_ANI}"
      SAMPLE_ROOT="${OUTDIR}/${DATASET}-${MODEL_TAG}-trained-ani-edmsampler-v2-seed-${START_SEED}"
      SAMPLE_SUB_PREFIX="fid-heun-${MODEL_TAG}-trained-ani-edmsampler-v2"
    fi

    echo "[INFO] EDM sampler: ${EDM_SAMPLER}" | tee -a "${LOGFILE}"

    # ---------------- EDM sampler ----------------
    for STEPS in "${STEPS_LIST[@]}"; do
      {
        echo "----------------------------------------"
        echo "[EDM SAMPLE] ${MODEL_TAG} | STEPS=${STEPS}"
        echo "----------------------------------------"
      } | tee -a "${LOGFILE}"

      "${PYTHON}" "${EDM_SAMPLER}" \
        --dataset "${DATASET}" \
        --model_tag "${MODEL_TAG}" \
        --ckpt_root "${CKPT_ROOT}" \
        --out "${OUTDIR}" \
        --batch "${BATCH}" \
        --start_seed "${START_SEED}" \
        --num_seeds "${NUM_SEEDS}" \
        --steps "${STEPS}" \
        2>&1 | tee -a "${LOGFILE}"

      SAMPLE_DIR="${SAMPLE_ROOT}/${SAMPLE_SUB_PREFIX}-${STEPS}"

      echo "[FID][EDM] ${MODEL_TAG} | STEPS=${STEPS}" | tee -a "${LOGFILE}"

      set +e
      torchrun --standalone --nproc_per_node=1 \
        "${FID_PY}" calc \
        --images="${SAMPLE_DIR}" \
        --ref="${FID_REF}" \
        2>&1 | tee -a "${LOGFILE}"
      set -e
    done
    
    # ---------------- STD sampler + FID ----------------
    echo "[INFO] Standard sampler" | tee -a "${LOGFILE}"

    for STEPS in "${STEPS_LIST[@]}"; do
      {
        echo "----------------------------------------"
        echo "[STD SAMPLE] ${MODEL_TAG} | STEPS=${STEPS}"
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

      # -------- STD sampler output dir --------
      SAMPLE_DIR="${OUTDIR}/${DATASET}-${MODEL_TAG}/fid-heun-${MODEL_TAG}-${STEPS}"

      echo "[FID][STD] ${MODEL_TAG} | STEPS=${STEPS}" | tee -a "${LOGFILE}"

      set +e
      torchrun --standalone --nproc_per_node=1 \
        "${FID_PY}" calc \
        --images="${SAMPLE_DIR}" \
        --ref="${FID_REF}" \
        2>&1 | tee -a "${LOGFILE}"
      set -e
    done

  done

  echo "[DONE] ${DATASET}" | tee -a "${LOGFILE}"

done

echo "ALL DATASETS DONE."
