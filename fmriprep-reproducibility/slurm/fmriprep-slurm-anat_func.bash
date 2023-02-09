#!/bin/bash

#SBATCH --array=1-5
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000M
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

DATASET=$1
PARTICIPANT=$2
SING_IMG=$3

INPUT_DIR=${SCRATCH}/fmriprep-lts
METHOD=ieee
if [ $(echo $SING_IMG | grep fuzzy) ] ; then
    METHOD=fuzzy
fi
OUTPUT_DIR=${INPUT_DIR}/outputs/${METHOD}/fmriprep_${DATASET}_${SLURM_ARRAY_TASK_ID}
export SINGULARITYENV_FS_LICENSE=${HOME}/.freesurfer.txt
export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow

ANAT_DATASET_1=${INPUT_DIR}/outputs/${METHOD}/fmriprep_${DATASET}_1_anat
ANAT_1_FINISHED=${ANAT_DATASET_1}/${DATASET}_1_${PARTICIPANT}_finished

module load singularity/3.6

# first remove anat flag if exists
if [ -f ${ANAT_1_FINISHED} ]; then
    rm ${ANAT_1_FINISHED}
fi

#copying input dataset into local scratch space
rsync -rltv --info=progress2 --exclude "outputs" --exclude "fmriprep-reproducibility" ${INPUT_DIR} ${SLURM_TMPDIR}

###
# First batch, anat only
###

singularity run --cleanenv -B ${SLURM_TMPDIR}/fmriprep-lts:/WORK -B ${HOME}/.cache/templateflow:/templateflow -B /etc/pki:/etc/pki/ \
    ${SLURM_TMPDIR}/fmriprep-lts/envs/singularity-images/${SING_IMG} \
    -w /WORK/fmriprep_work \
    --output-spaces MNI152NLin2009cAsym MNI152NLin6Asym \
    --anat-only --notrack --write-graph --resource-monitor \
    --omp-nthreads 1 --nprocs 1 --mem_mb 65536 \
    --participant-label ${PARTICIPANT} --random-seed 0 --skull-strip-fixed-seed \
    /WORK/inputs/openneuro/${DATASET} /WORK/inputs/openneuro/${DATASET}/derivatives/fmriprep participant
fmriprep_exitcode=$?

mkdir -p ${OUTPUT_DIR}_anat
scp -r ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work ${OUTPUT_DIR}_anat/fmriprep_${DATASET}-${PARTICIPANT}_${SLURM_ARRAY_TASK_ID}_anat.workdir
if [ $fmriprep_exitcode -eq 0 ] ; then
    scp -r ${SLURM_TMPDIR}/fmriprep-lts/inputs/openneuro/${DATASET}/derivatives/fmriprep/* ${OUTPUT_DIR}_anat
    rm -r ${SLURM_TMPDIR}/fmriprep-lts/inputs/openneuro/${DATASET}/derivatives/fmriprep/*
    scp ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work/fmriprep_wf/resource_monitor.json ${OUTPUT_DIR}_anat
    touch ${OUTPUT_DIR}_anat/${DATASET}_${SLURM_ARRAY_TASK_ID}_${PARTICIPANT}_finished
fi
rm -r ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work

###
# Second batch, func only (and anat stays the same among each iteration)
###

# wait for first anat iteration of current dataset to be ready, and then copy it
while [ ! -f ${ANAT_1_FINISHED} ]; do sleep 1; done
mkdir -p ${SLURM_TMPDIR}/fmriprep-lts/outputs/${METHOD}/
rsync -rltv --info=progress2 ${ANAT_DATASET_1} ${SLURM_TMPDIR}/fmriprep-lts/outputs/${METHOD}/

singularity run --cleanenv -B ${SLURM_TMPDIR}/fmriprep-lts:/WORK -B ${HOME}/.cache/templateflow:/templateflow -B /etc/pki:/etc/pki/ \
    ${SLURM_TMPDIR}/fmriprep-lts/envs/singularity-images/${SING_IMG} \
    -w /WORK/fmriprep_work \
    --output-spaces MNI152NLin2009cAsym MNI152NLin6Asym \
    --anat-derivatives /WORK/outputs/${METHOD}/fmriprep_${DATASET}_1_anat/fmriprep/${PARTICIPANT}/anat \
    --notrack --write-graph --resource-monitor \
    --omp-nthreads 1 --nprocs 1 --mem_mb 65536 \
    --participant-label ${PARTICIPANT} --random-seed 0 --skull-strip-fixed-seed \
    /WORK/inputs/openneuro/${DATASET} /WORK/inputs/openneuro/${DATASET}/derivatives/fmriprep participant
fmriprep_exitcode=$?

mkdir -p ${OUTPUT_DIR}_func
scp -r ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work ${OUTPUT_DIR}_func/fmriprep_${DATASET}-${PARTICIPANT}_${SLURM_ARRAY_TASK_ID}_func.workdir
if [ $fmriprep_exitcode -eq 0 ] ; then
    scp -r ${SLURM_TMPDIR}/fmriprep-lts/inputs/openneuro/${DATASET}/derivatives/fmriprep/* ${OUTPUT_DIR}_func
    rm -r ${SLURM_TMPDIR}/fmriprep-lts/inputs/openneuro/${DATASET}/derivatives/fmriprep/*
    scp ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work/fmriprep_wf/resource_monitor.json ${OUTPUT_DIR}_func
fi
rm -r ${SLURM_TMPDIR}/fmriprep-lts/fmriprep_work

exit $fmriprep_exitcode 
