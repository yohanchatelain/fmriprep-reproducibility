#!/bin/bash

MRI_STATS_PATH=/mnt/lustre/ychatel/fmriprep-reproducibility/

if [[ ${MRI_STATS_PATH} == "" ]]; then
    echo "MRI_STATS_PATH not set"
    exit 1
fi

SUCCESS_STATUS="success"
FAIL_STATUS="fail"

MRI_PICKLE_DIR='mri_pickle'
MRI_LOG='mri_log'

mkdir -p ${MRI_PICKLE_DIR} ${MRI_LOG}

confidences=(0.75 0.8 0.85 0.9 0.95 0.99 0.995)

function run_test() {
    TEST=$1
    REFERENCE_PREFIX=$2
    REFERENCE_DATASET=$3
    REFERENCE_SUBJECT=$4
    REFERENCE_NAME=$5
    TARGET_PREFIX=$6
    TARGET_DATASET=$7
    TARGET_SUBJECT=$8
    TARGET_NAME=$9
    STATUS=${10}
    FWH=${11}
    MASK=${12}

    echo " ### EXPECT PASS ### "
    echo "    TEST=$1"
    echo "    REFERENCE_PREFIX=$2"
    echo "    REFERENCE_DATASET=$3"
    echo "    REFERENCE_SUBJECT=$4"
    echo "    REFERENCE_NAME=$5"
    echo "    TARGET_PREFIX=$6"
    echo "    TARGET_DATASET=$7"
    echo "    TARGET_SUBJECT=$8"
    echo "    TARGET_NAME=$9"
    echo "    STATUS=${10}"
    echo "    FWH=${11}"
    echo "    MASK=${12}"

    PARALLEL=parallel.${RANDOM}
    for confidence in ${confidences[@]}; do
        OUTPUT="${TEST}_${confidence}_reference_${REFERENCE_NAME}_${REFERENCE_DATASET}_${REFERENCE_SUBJECT}_target_${TARGET_NAME}_${TARGET_DATASET}_${TARGET_SUBJECT}_fwh_${FWH}"
        echo "python3 ${MRI_STATS_PATH}/MRI-stats/__main__.py ${TEST} \
            --confidence ${confidence} \
            --reference-template MNI152NLin2009cAsym --data-type anat \
            --reference-prefix ${REFERENCE_PREFIX} --reference-dataset ${REFERENCE_DATASET} --reference-subject ${REFERENCE_SUBJECT} \
            --target-prefix ${TARGET_PREFIX} --target-dataset ${TARGET_DATASET} --target-subject ${TARGET_SUBJECT} \
            --mask-combination ${MASK} --smooth-kernel ${FWH} \
            --output ${MRI_PICKLE_DIR}/${OUTPUT}.pkl
            &> ${MRI_LOG}/${OUTPUT}.log" >>$PARALLEL
    done
    parallel -j 5 <$PARALLEL

    for confidence in ${confidences[@]}; do
        python3 ${MRI_STATS_PATH}/MRI-stats/mri_check_status.py --status=${STATUS} --filename="${MRI_PICKLE_DIR}/${OUTPUT}.pkl"
    done
}

function run_expect_pass() {
    TEST=$1
    REFERENCE_PREFIX=$2
    REFERENCE_DATASET=$3
    REFERENCE_SUBJECT=$4
    REFERENCE_NAME=$5
    TARGET_PREFIX=$6
    TARGET_DATASET=$7
    TARGET_SUBJECT=$8
    TARGET_NAME=$9
    FWH=${10}
    MASK=${11}

    echo " ### EXPECT PASS ### "
    echo "    TEST=$1"
    echo "    REFERENCE_PREFIX=$2"
    echo "    REFERENCE_DATASET=$3"
    echo "    REFERENCE_SUBJECT=$4"
    echo "    REFERENCE_NAME=$5"
    echo "    TARGET_PREFIX=$6"
    echo "    TARGET_DATASET=$7"
    echo "    TARGET_SUBJECT=$8"
    echo "    TARGET_NAME=$9"
    echo "    FWH=${10}"
    echo "    MASK=${11}"

    run_test $TEST $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME success $FWH $MASK
}

function parse_cmd() {
    cat >parse_cmd.py <<HERE
import json
import sys
import os

def parse_cmd(filename):
    with open(filename) as fi:
        return json.load(fi)


if '__main__' == __name__:

    filename = sys.argv[1]
    print(filename)
    if not (os.path.isfile(filename)):
        print(f'Unkown file {filename}')
        sys.exit(1)

    cmd = parse_cmd(filename)
    for dataset, labels in cmd.items():
        for label in labels.keys():
            print(f'{dataset} {label}')
HERE
    python3 parse_cmd.py $1 >inputs.txt
    rm -f parse_cmd.py
}

function run_all_subject() {
    parse_cmd fmriprep-cmd.json
    while read -r DATASET SUBJECT; do
        REFERENCE_DATASET=${DATASET}
        REFERENCE_SUBJECT=${SUBJECT}
        TARGET_DATASET=${DATASET}
        TARGET_SUBJECT=${SUBJECT}
        run_expect_pass $TEST $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME $FWH $MASK
    done <inputs.txt
}

TEST=$1
REFERENCE_PREFIX=$2
REFERENCE_NAME=$3
TARGET_PREFIX=$4
TARGET_NAME=$5
FWH=${SLURM_ARRAY_TASK_ID}
MASK=$6
REFERENCE_DATASET=$7
REFERENCE_SUBJECT=$8
TARGET_DATASET=$9
TARGET_SUBJECT=${10}

echo "### ARGS ###"
echo "TEST=${TEST}"
echo "REFERENCE_PREFIX=${REFERENCE_PREFIX}"
echo "REFERENCE_NAME=${REFERENCE_NAME}"
echo "TARGET_PREFIX=${TARGET_PREFIX}"
echo "TARGET_NAME=${TARGET_NAME}"
echo "FWH=${FWH}"
echo "MASK=${MASK}"

if [ -z $DATASET ]; then
    run_all_subject
elif [ -z $TARGET_DATASET ] ; then
    REFERENCE_DATASET=${REFERENCE_DATASET}
    REFERENCE_SUBJECT=${REFERENCE_SUBJECT}
    TARGET_DATASET=${REFERENCE_DATASET}
    TARGET_SUBJECT=${REFERENCE_SUBJECT}
    echo "REFERENCE_DATASET=${REFERENCE_DATASET}"
    echo "REFERENCE_SUBJECT=${REFERENCE_SUBJECT}"
    echo "TARGET_DATASET=${TARGET_DATASET}"
    echo "TARGET_SUBJECT=${TARGET_SUBJECT}"
    run_expect_pass $TEST \
		    $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME \
		    $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME \
		    $FWH $MASK
else
    REFERENCE_DATASET=${REFERENCE_DATASET}
    REFERENCE_SUBJECT=${REFERENCE_SUBJECT}
    TARGET_DATASET=${TARGET_DATASET}
    TARGET_SUBJECT=${TARGET_SUBJECT}
    echo "REFERENCE_DATASET=${REFERENCE_DATASET}"
    echo "REFERENCE_SUBJECT=${REFERENCE_SUBJECT}"
    echo "TARGET_DATASET=${TARGET_DATASET}"
    echo "TARGET_SUBJECT=${TARGET_SUBJECT}"
    run_expect_pass $TEST \
		    $REFERENCE_PREFIX $REFERENCE_DATASET $REFERENCE_SUBJECT $REFERENCE_NAME \
		    $TARGET_PREFIX $TARGET_DATASET $TARGET_SUBJECT $TARGET_NAME \
		    $FWH $MASK
fi
