import numpy as np
from statsmodels.stats.multitest import multipletests
import mri_printer
import warnings

# euler_constant = 0.5772156649015328


def pce(target, p_values, alpha):
    '''
    Compute the Per-Comparison Error rate (uncorrected)
    '''
    name = 'PCE'
    N = p_values.size
    threshold = alpha
    reject = p_values < threshold
    nb_reject = np.ma.sum(reject)
    ratio = reject/N

    if mri_printer.verbose:
        mri_printer.print_name_method('Per-Comparison Error (Uncorrected)')
        print(f'- Alpha                 = {threshold:f}')
        print(f'- Card(Reject)          = {nb_reject}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(Reject)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, nb_reject, N, alpha, name)

    return nb_reject, N


def pce_sig(target, ref, test, alpha):
    '''
    Compute the Per-Comparison Error rate (uncorrected) for significant bits
    '''
    name = 'PCE-sig'
    N = ref.size
    reject = test < ref
    nb_reject = np.ma.sum(reject)
    ratio = nb_reject/N

    if mri_printer.verbose:
        mri_printer.print_name_method('Per-Comparison Error (Uncorrected)')
        print(f'- Card(Reject)          = {nb_reject}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')
    mri_printer.print_result(target, nb_reject, N, alpha, name)

    return nb_reject, N


def mct(target, p_values, alpha, method, short_name, long_name):
    '''
    Generic method for compute Multiple Comparison Tests rate.
    '''
    name = short_name
    N = p_values.size

    reject = None
    corrected_threshold = None

    with warnings.catch_warnings():
        reject, _, corrected_threshold_sidak, corrected_threshold_bonferroni = multipletests(
            p_values, alpha=alpha, method=method, is_sorted=True)

        if method == 'bonferroni':
            corrected_threshold = corrected_threshold_bonferroni
        elif method == 'sidak':
            corrected_threshold = corrected_threshold_sidak

    nb_reject = np.ma.sum(reject)
    ratio = nb_reject / N

    if mri_printer.verbose:
        mri_printer.print_name_method(long_name)
        ct = corrected_threshold
        if ct is not None:
            print(f'- Alpha correction      = {ct:f} ({ct:.3e})')
        print(f'- Card(FP)              = {nb_reject}')
        print(f'- Card(Voxels)          = {N}')
        print(f'- Card(FP)/Card(Voxels) = {ratio:.2e} [{ratio*100:f}%]')

    mri_printer.print_result(target, nb_reject, N, alpha, name)

    return nb_reject, N


def fwe_bonferroni(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the Bonferonni correction
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='bonferroni',
               short_name='FWE-Bon',
               long_name='FWE (Bonferroni)')


def fwe_sidak(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the Sidak correction
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='sidak',
               short_name='FWE-Sidak',
               long_name='FWE (Sidak)')


def fwe_holm_sidak(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the Holm-Sidak correction
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='holm-sidak',
               short_name='FWE-HS',
               long_name='FWE (Holm-Sidak)')


def fwe_holm_bonferroni(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the Holm-Bonferonni correction
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='holm',
               short_name='FWE-HB',
               long_name='FWE (Holm-Bonferroni)')


def fwe_simes_hochberg(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the Simes-Hochberg correction
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='simes-hochberg',
               short_name='FWE-SH',
               long_name='FWE (Simes-Hochberg)')


def fdr_BH(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Benjamini-Hochberg)
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='fdr_bh',
               short_name='FDR-BH',
               long_name='FDR (Benjamini-Hochberg)')


def fdr_BY(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Benjamini-Yekutieli)
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='fdr_by',
               short_name='FDR-BY',
               long_name='FDR (Benjamini-Yekutieli)')


def fdr_TSBH(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Two-stage Benjamini-Hochberg)
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='fdr_tsbh',
               short_name='FDR-TSBH',
               long_name='FDR (Two-Stage Benjamini-Hochberg)')


def fdr_TSBY(target, p_values, alpha):
    '''
    Compute the failing voxels ratio using the False Discovery Rate correction (Two-Stage Benjamini-Yekutieli)
    '''
    return mct(target=target, p_values=p_values, alpha=alpha,
               method='fdr_tsbky',
               short_name='FDR-TSBY',
               long_name='FDR (Two-Stage Benjamini-Yekutieli)')
