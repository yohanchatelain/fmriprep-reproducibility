import logging

import numpy as np
from sklearn.model_selection import KFold

import stabilitest.pprinter as pprinter
import stabilitest.statistics
import stabilitest.statistics.multiple_testing as mri_mt
from stabilitest.collect import stats_collect
from stabilitest.MRI.mri_sample import MRISample


def perform_tests(methods, filename, confidences, *args, **info):
    """
    Compute the failing voxel ratio for the given target image
    for each method in methods
    Return a dictionnary with the method used a the key and the FVR as value
    """
    global_fp = {}

    def print_sep():
        return print(pprinter.sep_h3) if len(methods) > 1 else lambda: None

    for method in methods:
        for confidence in confidences:
            alpha = 1 - confidence
            if pprinter.verbose():
                print_sep()

            nb_reject, nb_test = method(filename, alpha, *args)
            stats_collect.append(
                **info,
                confidence=confidence,
                target=filename,
                reject=nb_reject,
                tests=nb_test,
                method=method.__name__,
            )
            
            global_fp[method.__name__] = nb_reject, nb_test
        print_sep()
    return global_fp


def compute_pvalues(
    args, target_id, reference_sample, target_sample, distribution, **info
):
    confidences = args.confidence
    sample_size = info["sample_size"]

    target_filename = target_sample.get_filenames(target_id)
    score_name = distribution.get_name()
    pprinter.print_info(score_name, sample_size, target_filename, target_id)

    # Get p-values and sort them into 1D array
    p_values = distribution.p_value(target_sample.get_subsample())
    p_values.sort()

    methods = mri_mt.get_methods(args)

    # Compute the failing-voxels ratio and store it into the global_fv dict
    fvr = perform_tests(methods, target_filename, confidences, p_values, **info)

    return fvr


def get_info(args, **extra_kwargs):
    reference_dataset = args.reference_dataset
    reference_subject = args.reference_subject
    reference_template = args.reference_template

    if "target_dataset" in args:
        target_dataset = (
            args.target_dataset if args.target_dataset else reference_dataset
        )
        target_subject = (
            args.target_subject if args.target_subject else reference_subject
        )
        target_template = (
            args.target_template if args.target_template else reference_template
        )
    else:
        target_dataset = reference_dataset
        target_subject = reference_subject
        target_template = reference_template

    fwhm = args.smooth_kernel
    mask = args.mask_combination

    return dict(
        reference_dataset=reference_dataset,
        reference_subject=reference_subject,
        reference_template=reference_template,
        target_dataset=target_dataset,
        target_subject=target_subject,
        target_template=target_template,
        fwhm=fwhm,
        mask=mask,
        **extra_kwargs,
    )


def compute_fvr_per_target(
    args,
    reference_sample,
    reference_ids,
    target_sample,
    target_ids,
    distribution,
    nb_round=None,
    kth_round=None,
):
    """
    Compute the failing-voxels ratio (FVR) for each target image in targets for the given methods.
    Args:
        @reference_sample: sample of reference
        @reference_ids: subsample ids
        @target_sample: sample of target
        @target_ids: subsample ids
    Return:
        A dictionnary that contains for each target the FVR for each method used.
    """

    sample_size = reference_sample.get_size()

    info = get_info(
        args, sample_size=sample_size, kth_round=kth_round, nb_round=nb_round
    )

    distribution.fit(reference_sample.get_subsample())
    if args.verbose:
        logging.info(f"Use {distribution.get_name()} model")

    fvr_per_target = {}
    for i in target_ids:
        fvr = compute_pvalues(
            args,
            target_id=i,
            reference_sample=reference_sample,
            target_sample=target_sample,
            **info,
        )
        fvr_per_target[target_sample.get_filenames(i)] = (fvr, None)

    return fvr_per_target


def compute_k_fold_fvr(args, reference_sample, target_sample, distribution, nb_rounds):
    """
    Compute the FVR by splitting the reference set in two training/testing sets.
    Do this k time by shuffling the training/testing sets.

    < Should give us an estimation of the FVR fluctuation.

    Return a list of FVR for each round
    """

    msg = f"{nb_rounds}-fold failing-voxels count"
    pprinter.print_sep1(f"{msg:^40}")

    kfold = KFold(nb_rounds)
    fvr_list = []

    def compute_k_fold_round(i, train_id, test_id):
        round_msg = f"Round {i}"
        pprinter.print_sep2(f"{round_msg:^40}")

        fvr = compute_fvr_per_target(
            args,
            reference_sample=reference_sample,
            reference_ids=train_id,
            target_sample=target_sample,
            target_ids=test_id,
            distribution=distribution,
            nb_round=nb_rounds,
            kth_round=i,
        )
        return fvr

    sample_ids = list(range(reference_sample.get_size()))
    fvr_list = [
        compute_k_fold_round(i, train_id, test_id)
        for i, (train_id, test_id) in enumerate(kfold.split(sample_ids), start=1)
    ]

    return fvr_list


def compute_all_include_fvr(args):
    sample = MRISample(args)

    distribution = stabilitest.statistics.distribution.get_distribution(args)

    sample.load_t1w(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        datatype=args.datatype,
    )
    sample.load_brain_mask(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        datatype=args.datatype,
    )

    fvr = compute_fvr_per_target(
        args,
        reference_sample=sample,
        reference_ids=list(range(sample.get_size())),
        target_sample=sample,
        target_ids=list(range(sample.get_size())),
        distribution=distribution,
        nb_round=1,
        kth_round=1,
    )

    return fvr


def compute_loo(args):
    sample = MRISample(args)
    sample.load_t1_and_brain_maks(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        datatype=args.datatype,
    )

    distribution = stabilitest.statistics.distribution.get_distribution(args)

    logging.info(f"Sample size: {sample.get_size()}")

    fvr = compute_k_fold_fvr(
        args,
        reference_sample=sample,
        target_sample=sample,
        distribution=distribution,
        nb_rounds=sample.get_size(),
    )

    return fvr


def compute_one_fvr(args):
    reference_sample = MRISample(args)
    reference_sample.load_t1_and_brain_maks(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        datatype=args.datatype,
    )

    target_sample = MRISample(args)
    target_sample.load_t1_and_brain_maks(
        prefix=args.target_prefix,
        subject=args.target_subject,
        dataset=args.target_dataset,
        template=args.target_template,
        datatype=args.datatype,
    )

    distribution = stabilitest.statistics.distribution.get_distribution(args)

    print(f"Sample size: {reference_sample.get_size()}")

    target_sample.resample(reference_sample.get_subsample(0))

    fvr = compute_fvr_per_target(
        args,
        reference_sample=reference_sample,
        reference_ids=list(range(reference_sample.get_size())),
        target_sample=target_sample,
        target_ids=list(range(target_sample.get_size())),
        distribution=distribution,
        nb_round=1,
        kth_round=1,
    )

    return fvr


def compute_k_fold(args):
    sample = MRISample(args)
    sample.load_t1_and_brain_maks(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        datatype=args.datatype,
    )

    distribution = stabilitest.statistics.distribution.get_distribution(args)

    logging.info(f"Sample size: {sample.get_size()}")

    fvr = compute_k_fold_fvr(
        args,
        reference_sample=sample,
        target_sample=sample,
        distribution=distribution,
        nb_rounds=args.k_fold_rounds,
    )

    return fvr


def compute_n_effective_over_voxels(alpha, p_values, N):
    """
    Compute the number of effective voxels from the voxels variances
    """
    # Compute the voxel-wise variances of the p_values
    var = np.var(p_values, axis=0, dtype=np.float64)
    var_mean = np.mean(var)
    var_std = np.std(var)
    neff = (alpha * (1 - alpha)) / var_mean
    f = var_mean / ((alpha * (1 - alpha)) / N)

    print(f"Mean (Var): {var_mean}")
    print(f"Std  (Var): {var_std}")
    print(f"N         : {N}")
    print(f"Neff      : {neff}")
    print(f"f         : {f}")
    print("=" * 30)


def compute_n_effective_over_rounds(test, alpha, phats, N):
    """
    Compute the number of effective voxels from the k FVR estimations
    """

    mean = np.mean(phats)
    var = np.var(phats)
    neff = (alpha * (1 - alpha)) / var
    f = var / ((alpha * (1 - alpha)) / N)

    print(f"Test      : {test}")
    print(f"Mean      : {mean}")
    print(f"Var       : {var}")
    print(f"Std       : {np.std(phats)}")
    print(f"N         : {N}")
    print(f"Neff      : {neff}")
    print(f"f         : {f}")
    print("-" * 30)


def compute_n_effective(alpha, phat_k_fold, N):
    phats_round = dict()
    p_values_voxel_across_rounds = []
    for rnd in phat_k_fold:
        for target, (phats, p_values) in rnd.items():
            p_values_voxel_across_rounds.append(p_values)
            for test, phat in phats.items():
                if v := phats_round.get(test, None):
                    v.append(phat)
                else:
                    phats_round[test] = [phat]

    # Compute N_eff by computing var_phat over voxels before avareging them
    compute_n_effective_over_voxels(alpha, p_values_voxel_across_rounds, N)

    for test, phats in phats_round.items():
        # Compute N_eff for each test over the k-rounds
        compute_n_effective_over_rounds(test, alpha, phats, N)
