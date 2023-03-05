import tqdm
import scipy
from significantdigits import Error, Method
import significantdigits
import nilearn.masking
import nibabel
import numpy as np
from sklearn.model_selection import KFold

import mri_multiple_testing as mri_mt
import mri_image
import mri_normality
import mri_printer
import mri_stats
import mri_gmm
from mri_collect import stats_collect
from itertools import chain
import multiprocessing


def compute_fvr(methods, target, confidences, *args, **info):
    '''
    Compute the failing voxel ratio for the given target image
    for each method in methods
    Return a dictionnary with the method used a the key and the FVR as value
    '''
    global_fp = {}

    def print_sep(): return print(mri_printer.sep_h3) if len(
        methods) > 1 else lambda: None

    for method in methods:
        for confidence in confidences:
            alpha = 1 - confidence
            if mri_printer.verbose:
                print_sep()

            nb_reject, nb_test = method(target, alpha, *args)
            stats_collect.append(**info,
                                 confidence=confidence,
                                 target=target.get_filename(),
                                 reject=nb_reject,
                                 tests=nb_test,
                                 method=method.__name__)
            global_fp[method.__name__] = nb_reject, nb_test
        print_sep()
    return global_fp


def compute_pvalues_stats(args, ith_target, target_T1, supermask,
                          parameters, weights,  methods, **info):

    confidences = args.confidence
    fwh = args.smooth_kernel
    sample_size = info['sample_size']

    # For each target image, compute the Z-score associated
    target_filename = target_T1.get_filename()
    target_masked = mri_image.get_masked_t1(
        target_T1, supermask, fwh, args.normalize)

    score_name, score_fun = mri_stats.get_score(args)
    mri_printer.print_info(score_name, sample_size,
                           target_filename, ith_target)

    # Turn Z-score into p-values and sort them into 1D array
    p_values = score_fun(args, target_masked, parameters, weights)
    p_values.sort()

    # Compute the failing-voxels ratio and store it into the global_fv dict
    fvr = compute_fvr(methods,
                      target_T1,
                      confidences,
                      p_values,
                      **info)

    return fvr


def compute_sig_stats(args,
                      ith_target,
                      references_T1,
                      target_T1,
                      supermask,
                      sig,
                      methods, **info):

    sig_error = significantdigits.Error.Relative
    sig_method = significantdigits.Method.General
    confidences = args.confidence
    fwh = args.smooth_kernel
    sample_size = info['sample_size']
    ref_sig = sig

    target_filename = target_T1.get_filename()
    target_masked = mri_image.get_masked_t1(
        target_T1, supermask, fwh, args.normalize)

    mri_printer.print_info('Sigbit', sample_size,
                           target_filename, ith_target)

    test_sig = significantdigits.significant_digits(references_T1,
                                                    reference=target_masked,
                                                    axis=0,
                                                    error=sig_error,
                                                    method=sig_method)

    fvr = compute_fvr(methods, target_T1, confidences,
                      ref_sig, test_sig, **info)

    return fvr


def sequential_fit_normal(X, fit):
    _iterable = tqdm.tqdm(iterable=range(X.shape[-1]),
                          total=X.shape[-1])
    _parameters = np.fromiter(chain.from_iterable(fit(
        X[..., i]) for i in _iterable), dtype=np.float64)
    parameters = dict(a=_parameters[..., 0],
                      loc=_parameters[..., 1],
                      scale=_parameters[..., 2])
    return parameters


def parallel_fit_normal(X, fit):
    # def func(i):
    #     return chain.from_iterable(fit(X[..., i]))

    # _iterable = tqdm.tqdm(iterable=range(X.shape[-1]),
    #                       total=X.shape[-1])

    _iterable = tqdm.tqdm(np.swapaxes(X, 0, 1))

    with multiprocessing.Pool() as pool:

        _parameters = np.fromiter(
            chain.from_iterable(pool.map(fit, _iterable, chunksize=500)), dtype=np.float64)

        parameters = dict(beta=_parameters[..., 0],
                          loc=_parameters[..., 1],
                          scale=_parameters[..., 2])
        return parameters

    return None


def skewnorm_fit(X):
    return chain.from_iterable(scipy.stats.skewnorm.fit(X))


def gennorm_fit(X):
    return chain.from_iterable(scipy.stats.gennorm.fit(X))


def compute_fvr_per_target(args, references_T1, targets_T1, supermask,
                           methods, nb_round=None, kth_round=None):
    '''
    Compute the failing-voxels ratio (FVR) for each target image in targets for the given methods.
    Args:
        @targets: list of target image encoded into Nifti1Image object
        @supermask: the union of the target brain masks into a numpy array
        @methods: the list of method to use to compute the FVR
    Return:
        A dictionnary that contains for each target the FVR for each method used.
    '''

    dataset = args.reference_dataset
    subject = args.reference_subject
    sample_size = len(references_T1)
    fwh = args.smooth_kernel

    info = dict(dataset=dataset,
                subject=subject,
                sample_size=sample_size,
                fwh=fwh,
                kth_round=kth_round,
                nb_round=nb_round)

    if args.gmm:
        if args.verbose:
            print("Use GMM model")
        gmm, _ = mri_gmm.gmm_fit(references_T1)
        mean = gmm.means_
        std = np.sqrt(gmm.covariances_)
        weights = gmm.weights_
    elif args.compare_significant_digits:
        if args.verbose:
            print("Compare significant digits")
        mean = np.mean(references_T1, axis=0)
        sig_error = significantdigits.Error.Relative
        sig_method = significantdigits.Method.General
        sig = significantdigits.significant_digits(references_T1,
                                                   reference=mean,
                                                   axis=0,
                                                   error=sig_error,
                                                   method=sig_method)
    elif args.gaussian_type == 'skew':
        if args.parallel_fitting:
            parameters = parallel_fit_normal(
                references_T1, scipy.stats.skewnorm.fit)
        else:
            parameters = sequential_fit_normal(
                references_T1, scipy.stats.skewnorm.fit)

    elif args.gaussian_type == 'general':
        if args.parallel_fitting:
            parameters = parallel_fit_normal(
                references_T1, scipy.stats.gennorm.fit)
        else:
            parameters = sequential_fit_normal(
                references_T1, scipy.stats.gennorm.fit)

    elif args.gaussian_type == 'normal':
        mean = np.mean(references_T1, axis=0)
        std = np.std(references_T1, axis=0)
        parameters = dict(loc=mean, scale=std)
        weights = 1

    fvr_per_target = {}
    for i, target_T1 in enumerate(targets_T1):

        if args.compare_significant_digits:

            fvr = compute_sig_stats(args,
                                    ith_target=i,
                                    references_T1=references_T1,
                                    target_T1=target_T1,
                                    supermask=supermask,
                                    sig=sig,
                                    methods=[mri_mt.pce_sig],
                                    **info)
        else:
            fvr = compute_pvalues_stats(args,
                                        ith_target=i,
                                        target_T1=target_T1,
                                        supermask=supermask,
                                        parameters=parameters,
                                        weights=weights,
                                        methods=methods,
                                        **info)

        fvr_per_target[target_T1.get_filename()] = (fvr, None)

    return fvr_per_target


def compute_k_fold_fvr(args, reference_T1, reference_mask, nb_rounds, methods):
    '''
    Compute the FVR by splitting the reference set in two training/testing sets.
    Do this k time by shuffling the training/testing sets.

    < Should give us an estimation of the FVR fluctuation.

    Return a list of FVR for each round
    '''

    mask_combination = args.mask_combination
    fwh = args.smooth_kernel

    msg = f'{nb_rounds}-fold failing-voxels count'
    mri_printer.print_sep1(f'{msg:^40}')

    kfold = KFold(nb_rounds)
    fvr_list = []

    def compute_k_fold_round(i, train_id, test_id):
        round_msg = f'Round {i}'
        mri_printer.print_sep2(f'{round_msg:^40}')

        train_t1 = reference_T1[train_id]
        train_mask = reference_mask[train_id]
        train_t1_masked, supermask = mri_image.mask_t1(
            train_t1, train_mask, mask_combination, fwh, args.normalize)

        test = reference_T1[test_id]

        fvr = compute_fvr_per_target(args,
                                     references_T1=train_t1_masked,
                                     targets_T1=test,
                                     supermask=supermask,
                                     methods=methods,
                                     nb_round=nb_rounds,
                                     kth_round=i)
        return fvr

    fvr_list = [compute_k_fold_round(i, train_id, test_id)
                for i, (train_id, test_id) in
                enumerate(kfold.split(reference_T1), start=1)]

    return fvr_list


def compute_all_include_fvr(args, methods):
    if args.verbose:
        print('In compute_all_include_fvr')

    # normality_mask_path = mri_normality.run_test_normality(args)
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type)

    reference_masked, supermask = mri_image.mask_t1(
        reference_t1s, reference_masks,
        args.mask_combination,
        args.smooth_kernel,
        args.normalize)

    fvr = compute_fvr_per_target(args,
                                 references_T1=reference_masked,
                                 targets_T1=reference_masked,
                                 supermask=supermask,
                                 methods=methods,
                                 nb_round=1,
                                 kth_round=1)

    return fvr


def compute_all_exclude_fvr(args, methods):
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    print(f'Sample size: {reference_sample_size}')

    fvr = compute_k_fold_fvr(args,
                             reference_T1=reference_t1s,
                             reference_mask=reference_masks,
                             nb_rounds=reference_sample_size,
                             methods=methods)

    return fvr


def compute_one_fvr(args, methods):
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    train_t1_masked, supermask = mri_image.mask_t1(reference_t1s,
                                                   reference_masks,
                                                   args.mask_combination,
                                                   args.smooth_kernel,
                                                   args.normalize)

    target_t1s, _ = mri_image.get_reference(
        prefix=args.target_prefix,
        subject=args.target_subject,
        dataset=args.target_dataset,
        template=args.target_template,
        data_type=args.data_type)

    print(f'Sample size: {reference_sample_size}')

    source_shape = reference_t1s[0].shape
    target_shape = target_t1s[0].shape

    if source_shape != target_shape:
        if args.verbose:
            print('Resampling target on reference')
            print('source shape', source_shape)
            print('target shape', target_shape)
        target_t1s = mri_image.resample_images(target_t1s, reference_t1s[0])

        print(target_t1s)
        for t in target_t1s:
            print('target shape', t.shape)

    fvr = compute_fvr_per_target(args,
                                 references_T1=train_t1_masked,
                                 targets_T1=target_t1s,
                                 supermask=supermask,
                                 methods=methods,
                                 nb_round=None,
                                 kth_round=None)

    return fvr


def compute_k_fold(args, methods):
    reference, supermask = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type)

    reference_sample_size = len(reference)
    nb_voxels_in_mask = np.count_nonzero(supermask)

    print(f'Sample size: {reference_sample_size}')

    fvr = compute_k_fold_fvr(args,
                             dataset=args.reference_dataset,
                             subject=args.reference_subject,
                             sample_size=reference_sample_size,
                             k=args.k_fold_rounds,
                             reference=reference,
                             supermask=supermask,
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
        template=args.reference_template,
        data_type=args.data_type)

    reference_sample_size = len(reference_t1s)

    print(f'Sample size: {reference_sample_size}')

    t1s_masked, supermask = mri_image.mask_t1(t1s=reference_t1s,
                                              masks=reference_masks,
                                              mask_combination=args.mask_combination,
                                              smooth_kernel=args.smooth_kernel,
                                              normalize=args.normalize)

    mean = np.mean(t1s_masked, axis=0)
    std = np.std(t1s_masked, axis=0)
    sig = significantdigits.significant_digits(array=t1s_masked,
                                               reference=mean,
                                               base=2,
                                               axis=0,
                                               error=Error.Relative,
                                               method=Method.General)

    filename = '_'.join([args.reference_prefix,
                         args.reference_dataset,
                         args.reference_subject,
                         args.reference_template,
                         args.mask_combination,
                         str(int(args.smooth_kernel))])

    def save(x, name):
        print(f'Unmask {filename}')
        x_img = nilearn.masking.unmask(x, supermask)
        print(f'Save NumPy {name}')
        np.save(f'{filename}_{name}', x)
        print(f'Save Niffi {name}')
        x_img.to_filename(f'{filename}_{name}.nii')

    save(mean, 'mean')
    save(std, 'std')
    save(sig, 'sig')
