import nilearn.masking
import nibabel
import numpy as np
from sklearn.model_selection import KFold

import mri_image
import mri_normality
import mri_printer
import mri_stats
from mri_collect import stats_collect


def compute_fvr(reference_dataset, reference_subject, reference_sample_size,
                target, p_values, alpha, methods, fwh,
                k=None, k_round=None):
    '''
    Compute the failing voxel ratio for the given target image
    for each method in methods
    Return a dictionnary with the method used a the key and the FVR as value
    '''
    global_fp = {}

    def print_sep(): return print(mri_printer.sep_h3) if len(
        methods) > 1 else lambda: None

    for method in methods:
        if mri_printer.verbose:
            print_sep()

        fp = method(target, p_values, alpha)
        stats_collect.append(dataset=reference_dataset,
                             subject=reference_subject,
                             confidence=1 - alpha,
                             sample_size=reference_sample_size,
                             target=target.get_filename(),
                             fwh=fwh,
                             fvr=fp,
                             method=method.__name__,
                             k_fold=k,
                             k_round=k_round)
        global_fp[method.__name__] = fp
    print_sep()
    return global_fp


def compute_fvr_per_target(dataset, subject, sample_size,
                           targets_T1, supermask,
                           mean, std,
                           alpha, fwh, score, methods, k=None, k_round=None):
    '''
    Compute the failing-voxels ratio (FVR) for each target image in targets for the given methods.
    Args:
        @targets: list of target image encoded into Nifti1Image object
        @supermask: the intersection of the target brain masks into a numpy array
        @mean: the mean of the reference brain images set into a numpy array
        @N:The number of voxels True inside the supermask
        @fuzzy_sample_size: The fuzzy sample size, equals to the reference set size
        @dof: The degree of freedom, taken as fuzzy_sample_size - 1
        @alpha: the 1 - confidence level
        @methods: the list of method to use to compute the FVR
    Return:
        A dictionnary that contains for each target the FVR for each method used.
    '''

    fvr_per_target = {}

    for i, target_T1 in enumerate(targets_T1):

        # For each target image, compute the Z-score associated
        target_filename = target_T1.get_filename()
        target_masked = mri_image.get_masked_t1(target_T1, supermask, fwh)

        score_name, score_fun = mri_stats.get_score(score=score)
        mri_printer.print_info(score_name, sample_size, target_filename, i)

        # Turn Z-score into p-values and sort them into 1D array
        p_values = score_fun(target_masked, mean, std)
        p_values.sort()

        # Compute the failing-voxels ratio and store it into the global_fv dict
        fvr = compute_fvr(reference_dataset=dataset,
                          reference_subject=subject,
                          reference_sample_size=sample_size,
                          target=target_T1,
                          p_values=p_values,
                          alpha=alpha,
                          methods=methods,
                          fwh=fwh,
                          k=k, k_round=k_round)

        fvr_per_target[target_T1.get_filename()] = (fvr, p_values)

        # Dump masked uncorrected map
        # mri_image.dump_failing_voxels(
        #     target_T1, target_mask, alpha, p_values, fwh)

    return fvr_per_target


def compute_k_fold_fvr(reference_dataset, reference_subject,
                       reference_T1, reference_mask,
                       mask_combination, fwh,
                       k, alpha, methods, score):
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
        train_t1 = reference_T1[train_id]
        train_sample_size = len(train_t1)
        train_mask = reference_mask[train_id]

        train_t1_masked, supermask = mri_image.mask_t1(
            train_t1, train_mask, mask_combination, fwh)
        mean = np.mean(train_t1_masked, axis=0)
        std = np.std(train_t1_masked, axis=0)

        test = reference_T1[test_id]
        fvr = compute_fvr_per_target(dataset=reference_dataset,
                                     subject=reference_subject,
                                     sample_size=train_sample_size,
                                     targets_T1=test,
                                     supermask=supermask,
                                     mean=mean,
                                     std=std,
                                     fwh=fwh,
                                     alpha=alpha,
                                     methods=methods,
                                     score=score,
                                     k=k,
                                     k_round=i)
        return fvr

    fvr_list = [compute_k_fold_round(i, train_id, test_id)
                for i, (train_id, test_id) in
                enumerate(kfold.split(reference_T1), start=1)]

    return fvr_list


# def compute_fvr_per_target(dataset, subject, sample_size, targets,
#                            supermask, means, stds, N, fuzzy_sample_size,
#                            dof, alpha, fwh, methods, k=None, k_round=None):


def compute_all_include_fvr(args, methods):
    if args.verbose:
        print('In compute_all_include_fvr')

    #normality_mask_path = mri_normality.run_test_normality(args)
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    print(f'Sample size: {reference_sample_size}')
    reference_masked, supermask = mri_image.mask_t1(
        reference_t1s, reference_masks,
        args.mask_combination,
        args.smooth_kernel)
    mean = np.mean(reference_masked, axis=0)
    std = np.std(reference_masked, axis=0)
    alpha = 1 - args.confidence

    fvr = compute_fvr_per_target(dataset=args.reference_dataset,
                                 subject=args.reference_subject,
                                 sample_size=reference_sample_size,
                                 targets_T1=reference_t1s,
                                 supermask=supermask,
                                 mean=mean,
                                 std=std,
                                 fwh=args.smooth_kernel,
                                 alpha=alpha,
                                 methods=methods,
                                 score=args.score)

    return fvr


def compute_all_include_gmm_fvr(args, methods):
    if args.verbose:
        print('In compute_all_include_gmm_fvr')

    gmm_reference = mri_image.get_reference_gmm(
        gmm_prefix=args.gmm_paths,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        n_components=args.gmm_component
    )

    normality_mask_path = mri_normality.run_test_normality(args)
    targets, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=normality_mask_path)

    reference_sample_size = len(targets)
    nb_voxels_in_mask = np.count_nonzero(supermask)
    dof = reference_sample_size - 1
    alpha = 1 - args.confidence

    gmmc = args.gmm_component

    means = []
    stds = []

    for i in range(gmmc):
        mean = np.full(gmm_reference['empty'].shape, 0)
        std = np.full(gmm_reference['empty'].shape, 0)
        for j, index in enumerate(gmm_reference['indices']):
            idx = tuple(index)
            ref = gmm_reference['model'][j]
            mean[idx] = ref.means_.ravel()[i]
            std[idx] = ref.covariances_.ravel()[i]
        means.append(mean)
        stds.append(np.sqrt(std))

    mean = mri_stats.get_mean_reference(targets)
    std = mri_stats.get_std_reference(targets)

    print(f'Shape ref mean: {mean.shape} | std: {std.shape}')
    print(f'Shape gmm mean: {means[0].shape} | std: {stds[0].shape}')
    print(f'mean Min norm {(mean-means[0]).min()}')
    print(f'mean Max norm {(mean-means[0]).max()}')
    print(f'std Min norm {(std-stds[0]).min()}')
    print(f'std Max norm {(std-stds[0]).max()}')

    filename = f'diff_{args.reference_dataset}_{args.reference_subject}_mean'
    mri_image.dump_image(filename, mean-means[0], targets[0].affine)
    mri_image.dump_image(filename, np.where(
        supermask, mean-means[0], 0), targets[0].affine)
    filename = f'diff_{args.reference_dataset}_{args.reference_subject}_std'
    mri_image.dump_image(filename, np.where(
        supermask, std-stds[0], 0), targets[0].affine)

    filename = f'gmm_mean_{args.reference_dataset}_{args.reference_subject}'
    mri_image.dump_image(filename, means[0], targets[0].affine)
    filename = f'mean_{args.reference_dataset}_{args.reference_subject}'
    mri_image.dump_image(filename, mean, targets[0].affine)
    filename = f'gmm_std_{args.reference_dataset}_{args.reference_subject}'
    mri_image.dump_image(filename, stds[0], targets[0].affine)
    filename = f'std_{args.reference_dataset}_{args.reference_subject}'
    mri_image.dump_image(filename, std, targets[0].affine)

    fvr = compute_fvr_per_target(
        dataset=args.reference_dataset,
        subject=args.reference_subject,
        sample_size=reference_sample_size,
        targets=targets,
        supermask=supermask,
        means=means,
        stds=stds,
        N=nb_voxels_in_mask,
        fuzzy_sample_size=reference_sample_size,
        dof=dof,
        alpha=alpha,
        methods=methods)

    return fvr


def compute_all_exclude_fvr(args, methods):
    # normality_mask_path = mri_normality.run_test_normality(args)
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    print(f'Sample size: {reference_sample_size}')
    alpha = 1 - args.confidence

    fvr = compute_k_fold_fvr(
        reference_dataset=args.reference_dataset,
        reference_subject=args.reference_subject,
        reference_T1=reference_t1s,
        reference_mask=reference_masks,
        mask_combination=args.mask_combination,
        k=reference_sample_size,
        fwh=args.smooth_kernel,
        alpha=alpha,
        methods=methods,
        score=args.score)

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
        means=[mean],
        stds=[std],
        N=nb_voxels_in_mask,
        fuzzy_sample_size=reference_sample_size,
        dof=dof,
        alpha=alpha,
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
        methods=methods)

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


def compute_stats(args):
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    print(f'Sample size: {reference_sample_size}')

    t1s_masked, supermask = mri_image.mask_t1(t1s=reference_t1s,
                                              masks=reference_masks,
                                              mask_combination=args.mask_combination,
                                              smooth_kernel=args.smooth_kernel)

    mean = np.mean(t1s_masked, axis=0)
    std = np.std(t1s_masked, axis=0)

    mean_img = nilearn.masking.unmask(mean, supermask)
    std_img = nilearn.masking.unmask(std, supermask)

    filename = '_'.join([args.reference_prefix,
                         args.reference_dataset,
                         args.reference_subject,
                         args.template,
                         args.mask_combination,
                         str(int(args.smooth_kernel))])

    np.save(f'{filename}_mean', mean)
    np.save(f'{filename}_std', std)

    mean_img.to_filename('{filename}_mean')
    std_img.to_filename('{filename}_std')
