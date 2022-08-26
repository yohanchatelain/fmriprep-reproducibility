import mri_fvr
from sklearn.model_selection import KFold
import mri_stats
import mri_image
import nibabel
import numpy as np
import mri_printer

from mri_collect import stats_collect
import mri_normality


def compute_fvr(dataset, subject, sample_size, target, p_values, alpha, nvoxels, methods, 
                k=None, k_round=None):
    '''
    Compute the failing voxel ratio for the given target image
    for each method in methods
    Return a dictionnary with the method used a the key and the FVR as value
    '''
    global_fp = dict()

    def print_sep(): return print(mri_printer.sep_h3) if len(
        methods) > 1 else lambda: None
    for method in methods:
        if mri_printer.verbose:
            print_sep()
        fp = method(target, p_values, alpha, nvoxels)
        stats_collect.append(dataset=dataset,
                             subject=subject,
                             confidence=1-alpha,
                             sample_size=sample_size,
                             target=target.get_filename(),
                             fvr=fp,
                             method=method.__name__,
                             k_fold=k,
                             k_round=k_round)
        global_fp[method.__name__] = fp
    print_sep()
    return global_fp


def compute_fvr_per_target(dataset, subject, sample_size, targets,
                           supermask, mean, std, N, fuzzy_sample_size,
                           dof, alpha, population, methods, k=None, k_round=None):
    '''
    Compute the failing-voxels ratio (FVR) for each target image in targets for the given methods.
    Args:
        @targets: list of target image encoded into Nifti1Image object
        @supermask: the intersection of the target brain masks into a numpy array
        @mean: the mean of the reference brain images set into a numpy array
        @std: the standard-deviation of the reference brain images set into a numpy array
        @N:The number of voxels True inside the supermask
        @fuzzy_sample_size: The fuzzy sample size, equals to the reference set size
        @dof: The degree of freedom, taken as fuzzy_sample_size - 1
        @alpha: the 1 - confidence level
        @methods: the list of method to use to compute the FVR
    Return:
        A dictionnary that contains for each target the FVR for each method used.
    '''

    def print_info(score, nsample, target):
        if mri_printer.verbose:
            name = f'{score} ({nsample} repetitions)'
            mri_printer.print_sep1(f'{name:^40}')
            header = f"Target ({i}): {target}"
            mri_printer.print_sep2(f'{header:^40}')

    def dump_failing_voxels(target, alpha, p_values, supermask):
        filename = target.get_filename().replace(
            '.nii.gz', f'_{alpha:.3f}_.nii.gz')
        print(f'dump failing voxels {filename}')
        fp_masked = np.logical_and(p_values < alpha, supermask)
        image = nibabel.Nifti1Image(
            np.where(fp_masked, target.get_fdata(), 0), target.affine)
        mri_image.dump_image(filename, image.get_fdata(), image.affine)

    fvr_per_target = dict()

    for i, target in enumerate(targets):

        # For each target image, compute the Z-score associated
        x = np.where(supermask, target.get_fdata(), 0)

        score_name, score = mri_stats.get_score(x, fuzzy_sample_size)
        print_info(score_name, fuzzy_sample_size, target.get_filename())

        # Turn Z-score into p-values and sort them into 1D array
        p_values = score(x, mean, std, dof)
        #dump_p_values(target, p_values, supermask, alpha)
        p_values_1d = p_values[supermask].copy().ravel()
        p_values_1d.sort()

        # Compute the failing-voxels ratio and store it into the global_fv dict
        # fvr = compute_fvr(target, p_values, alpha, N, methods)
        fvr = compute_fvr(dataset, subject, sample_size,
                          target, p_values_1d, alpha, N, methods, k=k, k_round=k_round)
        fvr_per_target[target.get_filename()] = (fvr, p_values)

        # Dump masked uncorrected map
        # dump_failing_voxels(target, alpha, p_values, supermask)

    return fvr_per_target


def compute_k_fold_fvr(dataset, subject, sample_size, k, reference, 
                       supermask, alpha, nb_voxels_in_mask, population, methods):
    '''
    Compute the FVR by splitting the reference set in two training/testing sets.
    Do this k time by shuffling the training/testing sets.

    < Should give us an estimation of the FVR fluctuation.

    Return a list of FVR for each round
    '''

    msg = f'{k}-fold failing-voxels count'
    mri_printer.print_sep1(f'{msg:^40}')

    kfold = KFold(k)
    fvr_list = []

    def compute_k_fold_round(i, train_id, test_id):
        round_msg = f'Round {i}'
        mri_printer.print_sep2(f'{round_msg:^40}')
        train = reference[train_id]
        test = reference[test_id]
        mean = mri_stats.get_mean_reference(train)
        std = mri_stats.get_std_reference(train)
        fuzzy_sample_size = len(train)
        dof = fuzzy_sample_size - 1
        fvr = compute_fvr_per_target(dataset, subject, sample_size,
                                     test, supermask,
                                     mean, std,
                                     nb_voxels_in_mask, fuzzy_sample_size,
                                     dof, alpha, population, methods, k=k, k_round=i)
        return fvr

    fvr_list = [compute_k_fold_round(i, train_id, test_id)
                for i, (train_id, test_id) in
                enumerate(kfold.split(reference), start=1)]

    return fvr_list


def compute_global_fvr(dataset, subject, sample_size, reference,
                       supermask, alpha, population, nb_voxels_in_mask,
                       methods):
    '''
    Use the set for the reference and the targets
    so the mean/std is computed using all the sample.
    Then each repetition is tested agains the reference.

    < We should pass the test for all repetition.
    '''
    mri_printer.print_sep1('Global failing voxels count')
    mean = mri_stats.get_mean_reference(reference)
    std = mri_stats.get_std_reference(reference)
    fuzzy_sample_size = len(reference)
    dof = fuzzy_sample_size - 1

    mri_image.dump_mean(reference[0], mean, supermask, alpha)
    mri_image.dump_std(reference[0], std, supermask, alpha)

    fvr = compute_fvr_per_target(dataset, subject, sample_size,
                                 reference, supermask,
                                 mean, std,
                                 nb_voxels_in_mask, fuzzy_sample_size,
                                 dof, alpha, population, methods)
    return fvr


def compute_all_include_fvr(args, methods):
    normality_mask_path = mri_normality.run_test_normality(args)
    reference, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=normality_mask_path)
    reference_sample_size = len(reference)
    nb_voxels_in_mask = np.count_nonzero(supermask)

    print(f'Sample size: {reference_sample_size}')
    mean = mri_stats.get_mean_reference(reference)
    std = mri_stats.get_std_reference(reference)
    dof = reference_sample_size - 1
    alpha = 1 - args.confidence

    fvr = compute_fvr_per_target(
        dataset=args.reference_dataset,
        subject=args.reference_subject,
        sample_size=reference_sample_size,
        targets=reference,
        supermask=supermask,
        mean=mean,
        std=std,
        N=nb_voxels_in_mask,
        fuzzy_sample_size=reference_sample_size,
        dof=dof,
        alpha=alpha,
        population=args.population,
        methods=methods)

    return fvr


def compute_all_exclude_fvr(args, methods):
    normality_mask_path = mri_normality.run_test_normality(args)
    reference, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=normality_mask_path)
    reference_sample_size = len(reference)
    nb_voxels_in_mask = np.count_nonzero(supermask)

    print(f'Sample size: {reference_sample_size}')
    alpha = 1 - args.confidence

    fvr = compute_k_fold_fvr(
        dataset=args.reference_dataset,
        subject=args.reference_subject,
        sample_size=reference_sample_size,
        k=reference_sample_size-1,
        reference=reference,
        supermask=supermask,
        alpha=alpha,
        nb_voxels_in_mask=nb_voxels_in_mask,
        population=args.population,
        methods=methods)

    return fvr


def compute_one_fvr(args, methods):
    normality_mask_path = mri_normality.run_test_normality(args)
    reference, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=normality_mask_path)

    reference_sample_size = len(reference)
    nb_voxels_in_mask = np.count_nonzero(supermask)
    print(f'Sample size: {reference_sample_size}')
    mean = mri_stats.get_mean_reference(reference)
    std = mri_stats.get_std_reference(reference)
    dof = reference_sample_size - 1
    alpha = 1 - args.confidence

    target = mri_image.get_target(
        target_prefix=args.target_prefix,
        target_subject=args.target_subject,
        target_dataset=args.target_dataset,
        template=args.template,
        data_type=args.data_type,
        normalize=args.normalize,
        normality_mask=normality_mask_path)

    fvr = compute_fvr_per_target(
        dataset=args.reference_dataset,
        subject=args.reference_subject,
        sample_size=reference_sample_size,
        targets=target,
        supermask=supermask,
        mean=mean,
        std=std,
        N=nb_voxels_in_mask,
        fuzzy_sample_size=reference_sample_size,
        dof=dof,
        alpha=alpha,
        population=args.population,
        methods=methods)

    return fvr


def compute_k_fold(args, methods):
    normality_mask_path = mri_normality.run_test_normality(args)
    reference, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=normality_mask_path)
    reference_sample_size = len(reference)
    nb_voxels_in_mask = np.count_nonzero(supermask)

    print(f'Sample size: {reference_sample_size}')
    alpha = 1 - args.confidence

    fvr = compute_k_fold_fvr(
        dataset=args.reference_dataset,
        subject=args.reference_subject,
        sample_size=reference_sample_size,
        k=args.k_fold_rounds,
        reference=reference,
        supermask=supermask,
        alpha=alpha,
        nb_voxels_in_mask=nb_voxels_in_mask,
        population=args.population,
        methods=methods)

    return fvr


def compute_global_fvr_no_recompute(dataset, subject, sample_size, reference, supermask, alpha, population, nb_voxels_in_mask, methods, mean, std, fuzzy_sample_size):
    '''
    Use the set for the reference and the targets
    so the mean/std is computed using all the sample.
    Then each repetition is tested agains the reference.

    < We should pass the test for all repetition.
    '''
    mri_printer.print_sep1(f'Global failing voxels count')
    dof = fuzzy_sample_size - 1

    # dump_mean(reference[0], mean, supermask, alpha)
    # dump_std(reference[0], std, supermask, alpha)

    fvr = compute_fvr_per_target(dataset, subject, sample_size,
                                 reference, supermask,
                                 mean, std,
                                 nb_voxels_in_mask, fuzzy_sample_size,
                                 dof, alpha, population, methods)
    return fvr


def compute_n_effective_over_voxels(alpha, p_values, N):
    '''
    Compute the number of effective voxels from the voxels variances
    '''
    # Compute the voxel-wise variances of the p_values
    var = np.var(p_values, axis=0, dtype=np.float64)
    var_mean = np.mean(var)
    var_std = np.std(var)
    neff = (alpha*(1-alpha)) / var_mean
    f = var_mean / ((alpha * (1-alpha))/N)

    print(f'Mean (Var): {var_mean}')
    print(f'Std  (Var): {var_std}')
    print(f'N         : {N}')
    print(f'Neff      : {neff}')
    print(f'f         : {f}')
    print('='*30)


def compute_n_effective_over_rounds(test, alpha, phats, N):
    '''
    Compute the number of effective voxels from the k FVR estimations
    '''

    mean = np.mean(phats)
    var = np.var(phats)
    neff = (alpha*(1-alpha)) / var
    f = var / ((alpha * (1-alpha))/N)

    print(f'Test      : {test}')
    print(f'Mean      : {mean}')
    print(f'Var       : {var}')
    print(f'Std       : {np.std(phats)}')
    print(f'N         : {N}')
    print(f'Neff      : {neff}')
    print(f'f         : {f}')
    print('-'*30)


def compute_n_effective(alpha, phat_k_fold, N):

    phats_round = dict()
    p_values_voxel_across_rounds = []
    for rnd in phat_k_fold:
        for target, (phats, p_values) in rnd.items():
            p_values_voxel_across_rounds.append(p_values)
            for test, phat in phats.items():
                if (v := phats_round.get(test, None)):
                    v.append(phat)
                else:
                    phats_round[test] = [phat]

    # Compute N_eff by computing var_phat over voxels before avareging them
    compute_n_effective_over_voxels(alpha, p_values_voxel_across_rounds, N)

    for test, phats in phats_round.items():
        # Compute N_eff for each test over the k-rounds
        compute_n_effective_over_rounds(test, alpha, phats, N)
