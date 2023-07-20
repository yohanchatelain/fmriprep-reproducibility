#!/usr/bin/env python3
# coding: utf-8

import stabilitest
import stabilitest.snr as snr
import warnings

import stabilitest.parse_args as parse_args
import stabilitest.MRI.mri_distance as mri_distance
import stabilitest.model as model
import mri_gmm
import mri_normality
import stabilitest.pprinter as pprinter
from stabilitest.collect import stats_collect

warnings.simplefilter(action="ignore", category=FutureWarning)


def run_all(args):
    fvr = model.run_all(args)
    return fvr


def run_loo(args):
    fvr = model.run_loo(args)
    return fvr


def run_one(args):
    fvr = model.run_one(args)
    return fvr


def run_normality(args):
    normality = mri_normality.run_test_normality(args)
    print(normality)


def run_k_fold(args):
    fvr = model.run_kfold(args)
    return fvr


def run_stats(args):
    stabilitest.statistics.stats.compute_stats(args)


def run_distance(args):
    mri_distance.main(args)


def run_snr(args):
    snr.main(args)


tests = {
    "all": run_all,
    "loo": run_loo,
    "k-fold": run_k_fold,
    "one": run_one,
    "normality": run_normality,
    "stats": run_stats,
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
