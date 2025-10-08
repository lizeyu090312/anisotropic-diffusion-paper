#!/bin/bash
set -e

# ==============================================================
# CONFIGURATION
# ==============================================================
DATASETS=("cifar10" "afhqv2" "ffhq")
TRAIN_SCRIPTS=("g_iso_train.py" "g_iso_train_discretize.py" "g_ani_train.py" "g_iso_wrapper_train.py" "g_ani_wrapper_train.py")
MODEL_TAGS=("g-iso" "g-iso-discretize" "g-ani" "g-iso-wrapper" "g-ani-wrapper")
OUTROOT="finetune"
SAMPLE_SCRIPT="g_sample.py"
FID_SCRIPT="fid.py"
NUM_IMAGES=50000
BATCH=1024

# Whether to keep all intermediate checkpoints or only the latest one
KEEP_CKPT=false   # set false to only keep latest checkpoint

# ==============================================================
# LEARNING RATES per (dataset, model)
# ==============================================================
declare -A LR_MAP
declare -A GLR_MAP

# ------------ CIFAR10 ------------
LR_MAP["cifar10:g-iso"]=1e-5
GLR_MAP["cifar10:g-iso"]=1e-2
LR_MAP["cifar10:g-iso-discretize"]=1e-5
GLR_MAP["cifar10:g-iso-discretize"]=1e-2
LR_MAP["cifar10:g-ani"]=2e-4
GLR_MAP["cifar10:g-ani"]=1e-3
GLR_MAP["cifar10:g-iso-wrapper"]=1e-3
GLR_MAP["cifar10:g-ani-wrapper"]=1e-1

# ------------ AFHQV2 ------------
LR_MAP["afhqv2:g-iso"]=1e-5
GLR_MAP["afhqv2:g-iso"]=1e-2
LR_MAP["afhqv2:g-iso-discretize"]=1e-5
GLR_MAP["afhqv2:g-iso-discretize"]=5e-2
LR_MAP["afhqv2:g-ani"]=1e-5
GLR_MAP["afhqv2:g-ani"]=1e-4
GLR_MAP["afhqv2:g-iso-wrapper"]=1e-2
GLR_MAP["afhqv2:g-ani-wrapper"]=1e-2

# ------------ FFHQ ------------
LR_MAP["ffhq:g-iso"]=1e-5
GLR_MAP["ffhq:g-iso"]=1e-2
LR_MAP["ffhq:g-iso-discretize"]=1e-5
GLR_MAP["ffhq:g-iso-discretize"]=1e-2
LR_MAP["ffhq:g-ani"]=1e-5
GLR_MAP["ffhq:g-ani"]=1e-4
GLR_MAP["ffhq:g-iso-wrapper"]=1e-2
GLR_MAP["ffhq:g-ani-wrapper"]=1e-2

# ==============================================================
# LOGGING SETUP
# ==============================================================
LOGDIR="logs"
mkdir -p "${LOGDIR}"

# ==============================================================
# MAIN LOOP
# ==============================================================
for DATASET in "${DATASETS[@]}"; do
    echo "=============================================="
    echo ">>> PROCESSING DATASET: ${DATASET}"
    echo "=============================================="

    # Choose step list
    if [ "$DATASET" == "cifar10" ]; then
        STEPS=(5 6 7 8 9 18 30 40)
    else
        STEPS=(5 6 7 8 10 20 40 60)
    fi

    # ----------------------------------------------------------
    # LOOP OVER MODELS
    # ----------------------------------------------------------
    for ((i=0; i<${#TRAIN_SCRIPTS[@]}; i++)); do
        TRAIN_SCRIPT=${TRAIN_SCRIPTS[$i]}
        MODEL_TAG=${MODEL_TAGS[$i]}
        KEY="${DATASET}:${MODEL_TAG}"
        LR=${LR_MAP[$KEY]:-"N/A"}  # wrapper will print "N/A"
        GLR=${GLR_MAP[$KEY]}

        LOGFILE="${LOGDIR}/${DATASET}_${MODEL_TAG}.log"
        : > "${LOGFILE}"  # clear previous log before writing new
        echo ">>> Logging everything to ${LOGFILE}"
        echo "=============================================="
        echo "Dataset: ${DATASET} | Model: ${MODEL_TAG}" | tee -a "${LOGFILE}"
        echo "lr=${LR}, glr=${GLR}" | tee -a "${LOGFILE}"
        echo "----------------------------------------------" | tee -a "${LOGFILE}"

        {
            echo ""
            echo ">>> TRAINING STAGE"
            echo "----------------------------------------------"

            # Wrapper models only have glr
            if [[ "${MODEL_TAG}" == *"wrapper"* ]]; then
                if [ "$KEEP_CKPT" = true ]; then
                    python ${TRAIN_SCRIPT} \
                        --dataset ${DATASET} \
                        --out ${OUTROOT} \
                        --batch ${BATCH} \
                        --glr ${GLR} \
                        --keep_all_ckpt
                else
                    python ${TRAIN_SCRIPT} \
                        --dataset ${DATASET} \
                        --out ${OUTROOT} \
                        --batch ${BATCH} \
                        --glr ${GLR}
                fi
            else
                if [ "$KEEP_CKPT" = true ]; then
                    python ${TRAIN_SCRIPT} \
                        --dataset ${DATASET} \
                        --out ${OUTROOT} \
                        --batch ${BATCH} \
                        --lr ${LR} \
                        --glr ${GLR} \
                        --keep_all_ckpt
                else
                    python ${TRAIN_SCRIPT} \
                        --dataset ${DATASET} \
                        --out ${OUTROOT} \
                        --batch ${BATCH} \
                        --lr ${LR} \
                        --glr ${GLR}
                fi
            fi

            echo ""
            echo ">>> SAMPLING STAGE"
            echo "----------------------------------------------"

            # Skip discretize model
            if [ "${MODEL_TAG}" != "g-iso-discretize" ]; then
                python ${SAMPLE_SCRIPT} \
                    --dataset ${DATASET} \
                    --model_tag ${MODEL_TAG} \
                    --out samples \
                    --steps "${STEPS[@]}" \
                    --num_seeds ${NUM_IMAGES} \
                    --batch ${BATCH}

                echo ""
                echo ">>> FID EVALUATION"
                echo "----------------------------------------------"

                for K in "${STEPS[@]}"; do
                    IMGDIR="samples/${DATASET}-${MODEL_TAG}/fid-heun-${MODEL_TAG}-${K}"
                    echo ">>> Calculating FID for ${IMGDIR}"

                    if [ "$DATASET" == "cifar10" ]; then
                        REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
                    elif [ "$DATASET" == "afhqv2" ]; then
                        REF="afhq/fid-refs/afhqv2-64x64.npz"
                    else
                        REF="ffhq/fid-refs/ffhq-64x64.npz"
                    fi

                    torchrun --standalone --nproc_per_node=1 ${FID_SCRIPT} calc \
                        --images=${IMGDIR} \
                        --num=${NUM_IMAGES} \
                        --ref="${REF}"
                done
            fi

            echo ""
            echo ">>> FINISHED MODEL ${MODEL_TAG} on ${DATASET}"
            echo "=============================================="
        } 2>&1 | tee -a "${LOGFILE}"
    done
done
