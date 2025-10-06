#!/bin/bash

# ==============================================================
# CONFIG
# ==============================================================
DATASETS=("cifar10" "afhqv2" "ffhq")
TRAIN_SCRIPTS=("g_iso_train.py" "g_iso_train_discretize.py" "g_ani_train.py" "g_iso_wrapper_train.py" "g_ani_wrapper_train.py")
MODEL_TAGS=("g-iso" "g-iso-discretize" "g-ani" "g-iso-wrapper" "g-ani-wrapper")
OUTROOT="finetune"
SAMPLE_SCRIPT="g_sample.py"
FID_SCRIPT="fid.py"
NUM_IMAGES=50000
BATCH=1024

# ==============================================================
# LOOP OVER DATASETS
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
    # 1. TRAIN STAGES
    # ----------------------------------------------------------
    for ((i=0; i<${#TRAIN_SCRIPTS[@]}; i++)); do
        TRAIN_SCRIPT=${TRAIN_SCRIPTS[$i]}
        MODEL_TAG=${MODEL_TAGS[$i]}

        echo ""
        echo ">>> TRAINING: ${MODEL_TAG} on ${DATASET}"
        echo "----------------------------------------------"
        python ${TRAIN_SCRIPT} --dataset ${DATASET} --out ${OUTROOT} --batch ${BATCH}
    done

    # ----------------------------------------------------------
    # 2. SAMPLING + FID  (skip g-iso-discretize)
    # ----------------------------------------------------------
    for MODEL_TAG in "g-iso" "g-ani" "g-iso-wrapper" "g-ani-wrapper"; do
        echo ""
        echo ">>> SAMPLING: ${MODEL_TAG} on ${DATASET}"
        echo "----------------------------------------------"

        # sample
        python ${SAMPLE_SCRIPT} \
            --dataset ${DATASET} \
            --model_tag ${MODEL_TAG} \
            --out samples \
            --steps "${STEPS[@]}" \
            --num_seeds ${NUM_IMAGES} \
            --batch ${BATCH}

        # FID evaluation
        for K in "${STEPS[@]}"; do
            IMGDIR="samples/${DATASET}-${MODEL_TAG}/fid-heun-${MODEL_TAG}-${K}"
            echo ">>> Calculating FID for ${IMGDIR}"

            if [ "$DATASET" == "cifar10" ]; then
                REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz"
            elif [ "$DATASET" == "afhqv2" ]; then
                REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz"
            else
                REF="https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz"
            fi

            torchrun --standalone --nproc_per_node=1 ${FID_SCRIPT} calc \
                --images=${IMGDIR} \
                --num=${NUM_IMAGES} \
                --ref="${REF}"
        done
    done
done

echo "=============================================="
echo ">>> ALL TRAINING AND SAMPLING COMPLETED!"
echo "=============================================="
