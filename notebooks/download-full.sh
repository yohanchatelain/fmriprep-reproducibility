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
# │   ├── 20.2.0                               : c6kg2
# │   ├── 20.2.1                               : jvd4z
# │   │   ├── Architecture                     : cmhkj
# │   │   ├── Random Rounding                  : scrnq
# │   │   ├── Random Rounding + Random Seed    : f6jwm
# │   │   ├── Random Seed                      : fk5d2
# │   │   └── Corrupted templates              : 2x5bw
# │   ├── 20.2.2                               : 9wtmb
# │   ├── 20.2.3                               : vckgz
# │   ├── 20.2.4                               : q5c2t
# │   ├── 20.2.5                               : nm2ac
# │   ├── 21.0.4                               : hxwpz
# │   ├── 22.1.1                               : qdhmv
# │   └── 23.0.0                               : 65wca
# └── significant-digits                       : vpe25

OSF_ROOT="7yez9"
OSF_RESULTS="ydqs6"
OSF_20_2_0="c6kg2"
OSF_20_2_1="jvd4z"
OSF_20_2_1_ARCH="cmhkj"
OSF_20_2_1_RANDROUND="scrnq"
OSF_20_2_1_RANDROUND_RANDSEED="f6jwm"
OSF_20_2_1_RANDSEED="fk5d2"
OSF_20_2_1_CORRUPTED="2x5bw"
OSF_20_2_2="9wtmb"
OSF_20_2_3="vckgz"
OSF_20_2_4="q5c2t"
OSF_20_2_5="nm2ac"
OSF_21_0_4="hxwpz"
OSF_22_1_1="qdhmv"
OSF_23_0_0="65wca"
OSF_SIGNIFICANT_DIGITS="vpe25"

osf -u $OSF_USERNAME -p $OSF_ROOT clone numerical-variability-data
# osf -u $OSF_USERNAME -p $OSF_RESULTS clone numerical-variability-data/fmriprep-fuzzylibm-results
mkdir numerical-variability-data/fmriprep-fuzzylibm-results
osf -u $OSF_USERNAME -p $OSF_20_2_0 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.0
osf -u $OSF_USERNAME -p $OSF_20_2_1 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1
osf -u $OSF_USERNAME -p $OSF_20_2_1_ARCH clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1/arch
osf -u $OSF_USERNAME -p $OSF_20_2_1_RANDROUND clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1/rr
osf -u $OSF_USERNAME -p $OSF_20_2_1_RANDROUND_RANDSEED clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1/rr.rs
osf -u $OSF_USERNAME -p $OSF_20_2_1_RANDSEED clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1/rs
osf -u $OSF_USERNAME -p $OSF_20_2_1_CORRUPTED clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.1/template
osf -u $OSF_USERNAME -p $OSF_20_2_2 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.2
osf -u $OSF_USERNAME -p $OSF_20_2_3 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.3
osf -u $OSF_USERNAME -p $OSF_20_2_4 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.4
osf -u $OSF_USERNAME -p $OSF_20_2_5 clone numerical-variability-data/fmriprep-fuzzylibm-results/20.2.5
osf -u $OSF_USERNAME -p $OSF_21_0_4 clone numerical-variability-data/fmriprep-fuzzylibm-results/21.0.4
osf -u $OSF_USERNAME -p $OSF_22_1_1 clone numerical-variability-data/fmriprep-fuzzylibm-results/22.1.1
osf -u $OSF_USERNAME -p $OSF_23_0_0 clone numerical-variability-data/fmriprep-fuzzylibm-results/23.0.0
osf -u $OSF_USERNAME -p $OSF_SIGNIFICANT_DIGITS clone numerical-variability-data/significant-digits