#!/usr/bin/env python3
# coding: utf-8

import stabilitest.snr as snr
import warnings

import stabilitest.parse_args as parse_args
import stabilitest.MRI.mri_distance as mri_distance
import stabilitest.mri_test as mri_test
import mri_gmm
import stabilitest.statistics.multiple_testing as mri_mt
import mri_normality
import stabilitest.pprinter as pprinter
from stabilitest.collect import stats_collect

warnings.simplefilter(action="ignore", category=FutureWarning)


def run_all_include(args):
    if args.gmm_paths or args.gmm_component:
        fvr = mri_test.compute_all_include_gmm_fvr(args)
    else:
        fvr = mri_test.compute_all_include_fvr(args)
    return fvr


def run_loo(args):
    fvr = mri_test.compute_loo(args)
    return fvr


def run_one(args):
    fvr = mri_test.compute_one_fvr(args)
    return fvr


def run_normality(args):
    normality = mri_normality.run_test_normality(args)
    print(normality)


def run_k_fold(args):
    fvr = mri_test.compute_k_fold(args)
    return fvr


def run_stats(args):
    mri_test.compute_stats(args)


def run_gmm(args):
    mri_gmm.main(args)


def run_distance(args):
    mri_distance.main(args)


def run_snr(args):
    snr.main(args)


tests = {
    "all-include": run_all_include,
    "loo": run_loo,
    "one": run_one,
    "normality": run_normality,
    "k-fold": run_k_fold,
    "stats": run_stats,
    "gmm": run_gmm,
    "distance": run_distance,
    "snr": run_snr,
}


def main():
    parser, parsed_args = parse_args.parse_args()
    if parsed_args.verbose:
        pprinter.enable_verbose_mode()

    if parsed_args.mri_test not in tests:
        parser.print_help()

    tests[parsed_args.mri_test](parsed_args)
    stats_collect.set_name(parsed_args.output)
    stats_collect.dump()


main()
