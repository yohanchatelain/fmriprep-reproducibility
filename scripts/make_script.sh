ROOT_PATH=${PWD}
REFERENCE_SEED=42
SIF_IMAGE=fmriprep-fuzzy-20.2.1.sif

read -r -d '' HELP <<EOM
usage: ./make_script.sh --root-path --reference-seed --sif-image \n
\t--root-path=path          Root path of the project\n
\t--reference-seed=seed     Seed used as reference\n
\t--sif-image=sif           Name of singularity image to run fmriprep\n
EOM

DEBUG=true
SUCCESS=0
FAILURE=1

function debug() {
    if $DEBUG; then
        echo >&2 "[DEBUG] " $*
    fi
}

function fail() {
    echo >&2 $*
    echo >&2 -e $HELP
    exit 1
}

function parse_internal_args() {
    case $1 in
    --root-path=*)
        ROOT_PATH=$(parse_arg $1)
        debug "ROOT_PATH=$ROOT_PATH"
        ;;
    --reference-seed=*)
        REFERENCE_SEED=$(parse_arg $1)
        debug "REFERENCE_SEED=$REFERENCE_SEED"
        ;;
    --sif-image=*)
        SIF_IMAGE=$(parse_arg $1)
        debug "SIF_IMAGE=$SIF_IMAGE"
        ;;
    --help)
        fail ""
        ;;
    *)
        fail "Unkown arg: ${1}"
        ;;
    esac
}

i=1
for arg in "$@"; do
    parse_internal_args $arg
    debug "Arg ${i}: ${arg}"
    let i++
done

debug "ROOT_PATH=$ROOT_PATH"
debug "REFERENCE_SEED=$REFERENCE_SEED"
debug "SIF_IMAGE=$SIF_IMAGE"

# Escaped path separator
echo ${ROOT_PATH} >.root_path
REGEXP_ROOT_PATH=$(sed 's=/=\\/=g' .root_path)
rm -f .root_path

debug "creating launch_fmriprep_fuzzy_all.sh"
sed "s/%%ROOT_PATH%%/${REGEXP_ROOT_PATH}/g" scripts/launch_fmriprep_fuzzy_all.sh.in >scripts/launch_fmriprep_fuzzy_all.sh

debug "creating launch_fmriprep_fuzzy.sh"
rm -f .regexp

echo "s/%%ROOT_PATH%%/${REGEXP_ROOT_PATH}/g" >>.regexp
echo "s/%%REFERENCE_SEED%%/${REFERENCE_SEED}/g" >>.regexp
echo "s/%%SIF_IMAGE%%/${SIF_IMAGE}/g" >>.regexp
sed -f .regexp scripts/launch_fmriprep_fuzzy.sh.in >scripts/launch_fmriprep_fuzzy.sh
