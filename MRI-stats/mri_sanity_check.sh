#!/bin/bash

SUCCESS_STATUS="success"
FAIL_STATUS="fail"

MRI_PICKLE_DIR='mri_pickle'
MRI_LOG='mri_log'

mkdir -p ${MRI_PICKLE_DIR} ${MRI_LOG}

confidences=(0.75 0.8 0.9 0.95 0.99 0.995)

function run_test_all_include() {
    REFERENCE_PREFIX=$1
    REFERENCE_DATASET=$2
    REFERENCE_SUBJECT=$3
    REFERENCE_NAME=$4
    STATUS=$5
    OUTPUT="all-include_${confidence}_reference_${REFERENCE_NAME}_${REFERENCE_DATASET}_${REFERENCE_SUBJECT}"
    rm -f run_parallel
    for confidence in ${confidences[@]}; do
        echo "python3 MRI-stats/__main__.py all-include \
            --confidence $confidence \
            --template MNI152NLin2009cAsym --data-type anat \
            --reference-prefix ${REFERENCE_PREFIX} --reference-dataset ${REFERENCE_DATASET} --reference-subject ${REFERENCE_SUBJECT} \
            --output ${OUTPUT} \
            &>${OUTPUT}.log" >>run_parallel
    done
    parallel -j $(nproc) <run_parallel
    python3 mri_check_status.py --status=${STATUS} --filename="${MRI_PICKLE_DIR}/${OUTPUT}.pkl"
}

function run_test_all_exclude() {
    REFERENCE_PREFIX=$1
    REFERENCE_DATASET=$2
    REFERENCE_SUBJECT=$3
    REFERENCE_NAME=$4
    STATUS=$5
    OUTPUT="all-exclude_${confidence}_reference_${REFERENCE_NAME}_${REFERENCE_DATASET}_${REFERENCE_SUBJECT}"
    rm -f run_parallel
    for confidence in ${confidences[@]}; do
        echo "python3 MRI-stats/__main__.py all-exclude \
            --confidence $confidence \
            --template MNI152NLin2009cAsym --data-type anat \
            --reference-prefix ${REFERENCE_PREFIX} --reference-dataset ${REFERENCE_DATASET} --reference-subject ${REFERENCE_SUBJECT} \
            --output ${OUTPUT} \
            &>${OUTPUT}.log" >>run_parallel
    done
    parallel -j $(nproc) <run_parallel
    python3 mri_check_status.py --status=${STATUS} --filename="${MRI_PICKLE_DIR}/${OUTPUT}.pkl"
}

function run_test_one() {
    REFERENCE_PREFIX=$1
    REFERENCE_DATASET=$2
    REFERENCE_SUBJECT=$3
    REFERENCE_NAME=$4
    TARGET_PREFIX=$5
    TARGET_DATASET=$6
    TARGET_SUBJECT=$7
    TARGET_NAME=$8
    STATUS=$9
    OUTPUT="one_${confidence}_reference_${REFERENCE_NAME}_${REFERENCE_DATASET}_${REFERENCE_SUBJECT}_target_${TARGET_NAME}_${TARGET_DATASET}_${TARGET_SUBJECT}"
    for confidence in ${confidences[@]}; do
        echo "python3 MRI-stats/__main__.py one \
            --confidence $confidence \
            --template MNI152NLin2009cAsym --data-type anat \
            --reference-prefix ${REFERENCE_PREFIX} --reference-dataset ${REFERENCE_DATASET} --reference-subject ${REFERENCE_SUBJECT} \
            --target-prefix ${TARGET_PREFIX} --target-dataset ${TARGET_DATASET} --target-subject ${TARGET_SUBJECT} \
            --output ${OUTPUT} \
            &>${OUTPUT}.log" >>run_parallel
    done
    parallel -j $(nproc) <run_parallel
    python3 mri_check_status.py --status=${STATUS} --filename="${MRI_PICKLE_DIR}/${OUTPUT}.pkl"
}

function run_expect_pass() {
    REFERENCE_PREFIX=$1
    REFERENCE_DATASET=$2
    REFERENCE_SUBJECT=$3
    REFERENCE_NAME=$4
    TARGET_PREFIX=$5
    TARGET_DATASET=$6
    TARGET_SUBJECT=$7
    TARGET_NAME=$8
    run_test_all_include $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $SUCCESS_STATUS
    run_test_all_exclude $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $SUCCESS_STATUS
    run_test_all_one $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME $SUCCESS_STATUS
}

REFERENCE_PREFIX=$1
REFERENCE_NAME=$2
TARGET_PREFIX=$3
TARGET_NAME=$4

# {
#   "ds000256": {
#     "sub-CTS201": [
#       "--participant-label CTS201"
#       ],
#     "sub-CTS210": [
#       "--participant-label CTS210"
#       ]
#   },
#   "ds001748": {
#     "sub-adult15": [
#       "--participant-label CTS201"
#       ],
#     "sub-adult16": [
#       "--participant-label CTS210"
#       ]
#   },
#   "ds002338": {
#     "sub-xp207": [
#       "--participant-label CTS201"
#       ],
#     "sub-xp201": [
#       "--participant-label CTS210"
#       ]
#   },
#     "ds001600": {
#     "sub-1": [
#       "--participant-label 1"
#       ]
#   },
#     "ds001771": {
#       "sub-36": [
#         "--participant-label sub-36"
#         ]
#   }

REFERENCE_DATASET=ds000256
REFERENCE_SUBJECT=sub-CTS201
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds000256
REFERENCE_SUBJECT=sub-CTS210
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds001748
REFERENCE_SUBJECT=sub-adult15
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds001748
REFERENCE_SUBJECT=sub-adult16
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds002338
REFERENCE_SUBJECT=sub-xp207
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds002338
REFERENCE_SUBJECT=sub-xp201
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME

REFERENCE_DATASET=ds001600
REFERENCE_SUBJECT=sub-1
run_expect_pass $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME
