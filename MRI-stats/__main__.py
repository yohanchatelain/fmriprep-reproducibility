#!/usr/bin/env python3
# coding: utf-8

import warnings

import mri_args
import mri_fvr
import mri_gmm
import mri_multiple_testing as mri_mt
import mri_normality
import mri_printer
from mri_collect import stats_collect

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_methods(args=None):
    return [mri_mt.pce,
            mri_mt.fdr_TSBY,
            mri_mt.fdr_TSBH,
            mri_mt.fdr_BY,
            mri_mt.fdr_BH,
            mri_mt.fwe_simes_hochberg,
            mri_mt.fwe_holm_bonferroni,
            mri_mt.fwe_holm_sidak,
            mri_mt.fwe_sidak,
            mri_mt.fwe_bonferroni]


methods = get_methods()


def run_all_include(args):
    if args.gmm_paths or args.gmm_component:
        fvr = mri_fvr.compute_all_include_gmm_fvr(args, methods)
    else:
        fvr = mri_fvr.compute_all_include_fvr(args, methods)
    return fvr


def run_all_exclude(args):
    fvr = mri_fvr.compute_all_exclude_fvr(args, methods)
    return fvr


def run_one(args):
    fvr = mri_fvr.compute_one_fvr(args, methods)
    return fvr


def run_normality(args):
    normality = mri_normality.run_test_normality(args)
    print(normality)


def run_k_fold(args):
    fvr = mri_fvr.compute_k_fold(args, methods)
    return fvr


def run_stats(args):
    mri_fvr.compute_stats(args)


def run_gmm(args):
    mri_gmm.main(args)


tests = {
    'all-include': run_all_include,
    'all-exclude': run_all_exclude,
    'one': run_one,
    'normality': run_normality,
    'k-fold': run_k_fold,
    'stats': run_stats,
    'gmm': run_gmm
}


def main():
    parser, args = mri_args.parse_args()
    if args.verbose:
        mri_printer.enable_verbose_mode()

    if args.mri_test not in test:
        parser.print_help()

    tests[args.mri_test](args)
    stats_collect.set_name(args.output)
    stats_collect.dump()


main()
