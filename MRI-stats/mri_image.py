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
import tqdm

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
    images = []
    for path in paths:
        image_path = glob.glob(os.path.join(path, preproc_re))
        if len(image_path) != 0:
            image = load_image(image_path[0])
        else:
            continue
        images.append(image)

    return np.array(images)


def get_masks(paths, brain_mask_re):
    '''
    Load Nifti1Image from mask paths
    '''
    masks = []
    for path in paths:
        mask_path = glob.glob(os.path.join(path, brain_mask_re))
        if len(mask_path) != 0:
            mask = load_image(mask_path[0])
            masks.append(mask)
        else:
            continue

    return np.array(masks)


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


def resample_image(source, target):
    return np.array(
        [nilearn.image.resample_to_img(source, target)]
    )


def resample_images(sources, target):
    return np.array(
        [nilearn.image.resample_to_img(source, target) for source in sources]
    )


def normalize_image(image):
    voxels = image.get_fdata()
    normalized_image = (voxels - voxels.min()) / (voxels.max() - voxels.min())
    new = nibabel.Nifti1Image(normalized_image, image.affine)
    new.set_filename(image.get_filename())
    return new


def get_preproc_re(subject, template,
                   preproc_ext=mri_constants.t1_preproc_extension):
    return f'{subject}_space-{template}{preproc_ext}'


def get_brainmask_re(subject, template,
                     brainmask_ext=mri_constants.brain_mask_extension):
    return f'{subject}_space-{template}{brainmask_ext}'


def get_paths(prefix, dataset, subject, data_type):
    regexp = os.path.join(
        prefix, f'*{dataset}*', 'fmriprep', subject, data_type)
    paths = glob.glob(regexp)
    return paths


def get_reference(prefix, subject, dataset, template, data_type,
                  reference_ext=mri_constants.t1_preproc_extension):
    '''
    Returns T1 + mask images for given prefix, subject and dataset
    '''
    preproc_re = get_preproc_re(subject, template, reference_ext)
    brain_mask_re = get_brainmask_re(subject, template)
    paths = get_paths(prefix, dataset, subject, data_type)

    images = get_images(paths, preproc_re)
    masks = get_masks(paths, brain_mask_re)

    if len(images) == 0:
        print('No T1 images found')
        raise Exception('T1ImagesEmpty')
    if len(masks) == 0:
        print('No brain masks found')
        raise Exception('BrainMasksEmpty')

    return images, masks


def get_masked_t1(t1, mask, smooth_kernel, normalize):
    if smooth_kernel == 0:
        smooth_kernel = None
    masked = nilearn.masking.apply_mask(imgs=t1,
                                        mask_img=mask,
                                        smoothing_fwhm=smooth_kernel)
    if normalize:
        masked = (masked - masked.min()) / (masked.max() - masked.min())

    return masked


def mask_t1(t1s, masks, mask_combination, smooth_kernel, normalize):
    supermask = combine_mask(masks, mask_combination)
    masked_t1s = map(lambda t1: get_masked_t1(
        t1, supermask, smooth_kernel, normalize), t1s)
    progress_bar = tqdm.tqdm(desc='Masking reference',
                             iterable=masked_t1s,
                             unit='image',
                             total=len(t1s),
                             )
    return np.stack(progress_bar), supermask


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


def get_template(template):
    image_path = tflow.get(template, desc=None, resolution=1,
                           suffix='T1w', extension='nii.gz')
    mask_path = tflow.get(template, desc="brain",
                          resolution=1, suffix="mask", extension='nii.gz')

    image = nibabel.load(image_path)
    mask = nibabel.load(mask_path)
    return mask_image(image, mask.get_fdata())
