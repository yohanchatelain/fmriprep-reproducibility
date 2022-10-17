import numpy as np

import mri_printer

euler_constant = 0.5772156649015328


def pce(target, p_values, alpha, N):
    '''
    Compute the Per-Comparison Error rate (uncorrected)
    '''
    name = 'PCE'
    threshold = alpha
    fp = p_values < threshold
    false_positive = np.ma.sum(fp)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('Per-Comparison Error (Uncorrected)')
        print(f'- Alpha                 = {threshold:f}')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio


def fwe_bonferroni(target, p_values, alpha, N):
    '''
    Compute the failing voxels ratio using the Bonferonni correction
    '''
    name = 'FWE-Bon'
    corrected_threshold = alpha / N
    fp = p_values < corrected_threshold
    false_positive = np.ma.sum(fp)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('FWE (Bonferroni)')
        print(
            f'- Alpha correction      = {corrected_threshold:f} ({corrected_threshold:.3e})')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio


def fwe_holm_bonferroni(target, p_values, alpha, N):
    '''
    Compute the failing voxels ratio using the Holm-Bonferonni correction
    '''
    name = 'FWE-HB'
    thresholds = alpha / (N+1-np.arange(1, N+1))
    fp = np.where(p_values < thresholds)[0]
    corrected_threshold_index = np.max(fp) if fp.size != 0 else 0
    corrected_threshold = p_values[corrected_threshold_index]
    false_positive = np.ma.sum(p_values < corrected_threshold)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('FWE (Holm-Bonferroni)')
        print(
            f'- Alpha correction      = {corrected_threshold:f} ({corrected_threshold:.3e})')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio


def fdr_BH(target, p_values, alpha, N):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Benjamini-Hochberg)
    '''
    name = 'FDR-BH'
    thresholds = alpha * np.arange(1, N+1) / N
    fp = np.where(p_values < thresholds)[0]
    corrected_threshold_index = np.max(fp) if fp.size != 0 else 0
    corrected_threshold = p_values[corrected_threshold_index]
    false_positive = np.ma.sum(p_values < corrected_threshold)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('FDR (Benjamini-Hochberg)')
        print(
            f'- Alpha correction      = {corrected_threshold:f} ({corrected_threshold:.3e})')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio


def fdr_BY(target, p_values, alpha, N):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Benjamini-Yekutieli)
    '''
    name = 'FDR-BY'
    c = np.log(N) + euler_constant
    thresholds = alpha * np.arange(1, N+1) / (N * c)
    fp = np.where(p_values < thresholds)[0]
    corrected_threshold_index = np.max(fp) if fp.size != 0 else 0
    corrected_threshold = p_values[corrected_threshold_index]
    false_positive = np.ma.sum(p_values < corrected_threshold)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('FDR (Benjamini-Yekutieli)')
        print(
            f'- Alpha correction      = {corrected_threshold:f} ({corrected_threshold:.3e})')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio


def fdr_storey(target, p_values, alpha, N):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Storey)
    FDR <= pi_0 * alpha, with pi_0^-1 ~ H(p)
        H(p) = (1-lambda)*N / (sum( I(p_i < lambda) ) + 1)
        I(x) = 1 if x == true else 0
    standard choice of lambda = 1/2
    '''
    _lambda = 0.5
    name = f'FDR-S-{_lambda}'

    _sum_H = np.ma.sum(p_values > _lambda)
    H = ((1 - _lambda) * N) / (_sum_H + 1)
    thresholds = alpha * H * np.arange(1, N+1) / N
    fp = np.where(p_values < thresholds)[0]
    corrected_threshold_index = np.max(fp) if fp.size != 0 else 0
    corrected_threshold = p_values[corrected_threshold_index]
    false_positive = np.ma.sum(p_values < corrected_threshold)
    ratio = false_positive/N

    if mri_printer.verbose:
        mri_printer.print_name_method('FDR Storey-{_lambda}')
        print(
            f'- Alpha correction      = {corrected_threshold:f} ({corrected_threshold:.3e})')
        print(f'- Card(FP)              = {false_positive}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, ratio, alpha, name)

    return ratio
    