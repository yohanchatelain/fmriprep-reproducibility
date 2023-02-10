import pickle
from templateflow import api as tflow
import nibabel
import numpy as np
import glob
import os
import nilearn
import nilearn.masking

import mri_printer as mrip
import mri_constants
import mri_normality

'''
Loader and dumper
'''


def load_image(path):
    return nibabel.load(path)


def mask_image(image, mask):
    masked_image = np.ma.where(mask, image.get_fdata(), 0)
    return nibabel.Nifti1Image(masked_image, image.affine)


def dump_image(filename, x, affine):
    image = nibabel.Nifti1Image(x, affine)
    nibabel.save(image, filename)


def dump_stat(target, stat_image, supermask, alpha, stat_name=''):
    '''
    Generic dumping function
    '''
    filename = target.get_filename().replace(
        '.nii.gz', f'_{stat_name}_{alpha:.3f}_.nii.gz')
    mrip.print_debug(f'Dump {stat_name} {filename}')
    masked_image = np.where(supermask, stat_image, 0)
    image = nibabel.Nifti1Image(masked_image, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_failing_voxels(target, mask, alpha, p_values, fwh):
    confidence = 1 - alpha
    new_filename = f'_{confidence}_fwh_{fwh}.nii.gz'
    filename = target.get_filename().replace('.nii.gz', new_filename)
    mask = mask.get_fdata().astype('bool')
    fp_masked = np.logical_and(p_values <= alpha, mask)
    image = nibabel.Nifti1Image(fp_masked, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_mean(target, mean, supermask, alpha):
    '''
    Dump mean for masked target
    '''
    dump_stat(target, mean, supermask, alpha, stat_name='mean')


def dump_std(target, std, supermask, alpha):
    '''
    Dump std for masked target
    '''
    dump_stat(target, std, supermask, alpha, stat_name='std')


def dump_p_values(target, p_value, supermask, alpha):
    '''
    Dump p-values (with alpha threshold) for masked target 
    '''
    dump_stat(target, p_value, supermask, alpha, stat_name='p_value')


def get_images(paths, preproc_re):
    '''
    Load Nifti1Image from image paths
    '''
    return np.array([load_image(glob.glob(os.path.join(path, preproc_re))[0])
                     for path in paths])


def get_masks(paths, brain_mask_re):
    '''
    Load Nifti1Image from mask paths
    '''
    return np.array([load_image(glob.glob(os.path.join(path, brain_mask_re))[0])
                     for path in paths])


def combine_mask(masks_list, operator):
    '''
    Combine mask depending on the operator
    '''
    if operator == 'union':
        threshold = 0
    elif operator == 'intersection':
        threshold = 1
    else:
        threshold = 0.5
    # print(masks_list)
    return nilearn.masking.intersect_masks(masks_list, threshold=threshold)


def smooth_image(image, kernel_smooth):
    '''
    Smooth image with kernel size kernel_smooth
    '''
    if kernel_smooth > 0:
        return nilearn.image.smooth_img(image, kernel_smooth)
    else:
        return image


def normalize_image(image):
    normalized_image = image.get_fdata() / image.get_fdata().max()
    return nibabel.Nifti1Image(normalized_image, image.affine)


def get_reference_image(image, mask, normalize=None, smooth_kernel=None,
                        normality_mask=None):
    '''
    Get one image from the reference sample
    Apply supermask, normality_mask and normalization if required
    '''

    if normality_mask is not None:
        mask = np.ma.logical_and(mask, ~normality_mask)
    # Always smooth before masking to avoid blurried borders
    masked_image = normalize_image(image) if normalize else image
    masked_image = smooth_image(image, smooth_kernel)
    masked_image = mask_image(masked_image, mask)
    masked_image.set_filename(image.get_filename())
    return masked_image


def get_preproc_re(subject, template,
                   preproc_ext=mri_constants.preproc_extension):
    return f'{subject}*{template}{preproc_ext}'


def get_brainmask_re(subject, template,
                     brainmask_ext=mri_constants.brain_mask_extension):
    return f'{subject}*{template}{brainmask_ext}'


def get_paths(prefix, dataset, subject, data_type):
    regexp = os.path.join(
        prefix, f'*{dataset}*', 'fmriprep', subject, data_type)
    paths = glob.glob(regexp)
    return paths


def get_reference(prefix, subject, dataset, template, data_type):
    '''
    Returns T1 + mask images for given prefix, subject and dataset
    '''
    preproc_re = get_preproc_re(subject, template)
    brain_mask_re = get_brainmask_re(subject, template)
    paths = get_paths(prefix, dataset, subject, data_type)

    images = get_images(paths, preproc_re)
    masks = get_masks(paths, brain_mask_re)

    return images, masks


def get_masked_t1(t1, mask, smooth_kernel):
    if smooth_kernel == 0:
        smooth_kernel = None
    return nilearn.masking.apply_mask(imgs=t1,
                                      mask_img=mask,
                                      smoothing_fwhm=smooth_kernel)


def mask_t1(t1s, masks, mask_combination, smooth_kernel):
    supermask = combine_mask(masks, mask_combination)
    masked_t1s = map(lambda t1: get_masked_t1(
        t1, supermask, smooth_kernel), t1s)
    return np.stack(masked_t1s), supermask


def get_reference_gmm(gmm_prefix,
                      reference_subject,
                      reference_dataset,
                      n_components):
    path = f'{reference_dataset}_{reference_subject}_AI_{n_components}'
    if gmm_prefix:
        path = gmm_prefix + os.path.sep + path
    with open(path, 'rb') as fi:
        return pickle.load(fi)
    return None


def get_reference_args(args):
    return get_reference(subject=args.subject,
                         template=args.template,
                         reference=args.reference,
                         dataset=args.dataset,
                         data_type=args.data_type,
                         mask_combination=args.mask_combination,
                         normalize=args.normalize,
                         smooth_kernel=args.smooth_kernel,
                         normality_mask=args.normality_mask)


def get_target_image(path, preproc_re, brain_mask_re,
                     normalize, normality_mask):
    image_path = glob.glob(os.path.join(path, preproc_re))[0]
    brain_path = glob.glob(os.path.join(path, brain_mask_re))[0]
    image = load_image(image_path)
    brain_mask = load_image(brain_path)
    if normality_mask is None:
        mask = brain_mask
    else:
        mask = np.ma.logical_and(brain_mask, ~normality_mask)
    normalized_image = normalize_image(image) if normalize else image
    masked_image = mask_image(normalized_image, mask)
    masked_image.set_filename(image.get_filename())
    return masked_image


def get_target(target_prefix, target_subject, target_dataset,
               template, data_type, normalize, normality_mask):
    # Mask where True values are voxels failings Shapiro-Wilk test
    normality_mask = mri_normality.get_normality_mask(
        normality_mask_dir=normality_mask)
    preproc_re = get_preproc_re(target_subject, template)
    brain_mask_re = get_brainmask_re(target_subject, template)
    paths = get_paths(target_prefix, target_dataset, target_subject, data_type)
    data = [get_target_image(path, preproc_re, brain_mask_re,
                             normalize, normality_mask)
            for path in paths]
    return np.array(data)


def get_target_args(args):
    return get_target(target_prefix=args.target,
                      target_subject=args.subject_target,
                      template=args.template,
                      target_dataset=args.dataset_target,
                      data_type=args.data_type,
                      normalize=args.normalize,
                      normality_mask=args.normality_mask)


def get_template(template):
    image_path = tflow.get(template, desc=None, resolution=1,
                           suffix='T1w', extension='nii.gz')
    mask_path = tflow.get(template, desc="brain",
                          resolution=1, suffix="mask", extension='nii.gz')

    image = nibabel.load(image_path)
    mask = nibabel.load(mask_path)
    return mask_image(image, mask.get_fdata())
