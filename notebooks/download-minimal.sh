#!/bin/bash

# Check OSF environment variables are set
if [ -z "$OSF_TOKEN" ]; then
    echo "OSF_TOKEN is not set"
    exit 1
fi
if [ -z "$OSF_USERNAME" ]; then
    echo "OSF_USERNAME is not set"
    exit 1
fi

# Download data from OSF

# .                                            : 7yez9
# ├── fmriprep-fuzzylibm-results               : ydqs6
# └── significant-digits                       : vpe25

OSF_ROOT="7yez9"
OSF_SIGNIFICANT_DIGITS="vpe25"

osf -u $OSF_USERNAME -p $OSF_ROOT clone numerical-variability-data
mkdir numerical-variability-data/fmriprep-fuzzylibm-results
osf -u $OSF_USERNAME -p $OSF_SIGNIFICANT_DIGITS clone numerical-variability-data/significant-digits

# Extract archive

for f in $(find numerical-variability-data -name "*.tar.gz"); do
    echo "Extracting $f"
    tar -C $(dirname $f) -xf $f
done