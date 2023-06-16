import pickle
from templateflow import api as tflow
import nibabel
import numpy as np
import glob
import os
import nilearn
import nilearn.masking
from multiprocessing import Pool
import logging

import stabilitest.pprinter as mrip
import stabilitest.MRI.mri_constants as mri_constants
import mri_normality
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

"""
Loader and dumper
"""


def load_derivative(path):
    return nibabel.load(path)


def mask_image(image, mask):
    masked_image = np.ma.where(mask, image.get_fdata(), 0)
    return nibabel.Nifti1Image(masked_image, image.affine)


def dump_image(filename, x, affine):
    image = nibabel.Nifti1Image(x, affine)
    nibabel.save(image, filename)


def dump_stat(target, stat_image, supermask, alpha, stat_name=""):
    """
    Generic dumping function
    """
    filename = target.get_filename().replace(
        ".nii.gz", f"_{stat_name}_{alpha:.3f}_.nii.gz"
    )
    mrip.print_debug(f"Dump {stat_name} {filename}")
    masked_image = np.where(supermask, stat_image, 0)
    image = nibabel.Nifti1Image(masked_image, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_failing_voxels(target, mask, alpha, p_values, fwh):
    confidence = 1 - alpha
    new_filename = f"_{confidence}_fwh_{fwh}.nii.gz"
    filename = target.get_filename().replace(".nii.gz", new_filename)
    mask = mask.get_fdata().astype("bool")
    fp_masked = np.logical_and(p_values <= alpha, mask)
    image = nibabel.Nifti1Image(fp_masked, target.affine)
    dump_image(filename, image.get_fdata(), image.affine)


def dump_mean(target, mean, supermask, alpha):
    """
    Dump mean for masked target
    """
    dump_stat(target, mean, supermask, alpha, stat_name="mean")


def dump_std(target, std, supermask, alpha):
    """
    Dump std for masked target
    """
    dump_stat(target, std, supermask, alpha, stat_name="std")


def dump_p_values(target, p_value, supermask, alpha):
    """
    Dump p-values (with alpha threshold) for masked target
    """
    dump_stat(target, p_value, supermask, alpha, stat_name="p_value")


def load_derivatives(root_paths, derivative_path):
    """
    Load derivatives from paths
    """
    derivatives = []
    for path in root_paths:
        dpath = glob.glob(os.path.join(path, derivative_path))
        if len(dpath) > 1:
            logging.warning(f"More than one file found for {dpath}")
            image = load_derivative(dpath[0])
        else:
            continue
        derivatives.append(image)

    return np.array(derivatives)


def combine_mask(masks_list, operator):
    """
    Combine mask depending on the operator
    """
    if operator == "union":
        threshold = 0
    elif operator == "intersection":
        threshold = 1
    else:
        threshold = 0.5
    return nilearn.masking.intersect_masks(masks_list, threshold=threshold)


def resample_image(source, target):
    return np.array([nilearn.image.resample_to_img(source, target)])


def resample_images(sources, target):
    resampled_images = []
    for source in sources:
        resampled = nilearn.image.resample_to_img(source, target)
        resampled.set_filename(source.get_filename())
        resampled_images.append(resampled)
    return np.array(resampled_images)


def normalize_ndarray(ndarray):
    return (ndarray - ndarray.min()) / (ndarray.max() - ndarray.min())


def normalize_image(image):
    voxels = image.get_fdata()
    normalized_image = normalize_ndarray(voxels)
    new = nibabel.Nifti1Image(normalized_image, image.affine)
    new.set_filename(image.get_filename())
    return new


def get_paths(prefix, dataset):
    regexp = os.path.join(prefix, f"*{dataset}*", "**")
    paths = glob.glob(regexp, recursive=True)
    return paths


def load(
    prefix,
    dataset,
    subject,
    template,
    datatype,
    derivative,
):
    """
    Returns T1 + mask images for given prefix, subject and dataset
    """
    root_paths = get_paths(prefix, dataset)
    derivative_path = mri_constants.get_derivatives_path(
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=derivative,
    )

    derivatives = load_derivatives(root_paths, derivative_path)

    if len(derivatives) == 0:
        print(f"No derivatives {derivative} found")
        raise Exception("DerivativeEmpty")

    return derivatives


def load_t1w(prefix, dataset, subject, template, datatype):
    return load(
        prefix=prefix,
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=mri_constants.Derivative.T1wPreproc,
    )


def load_brain_mask(prefix, dataset, subject, template, datatype):
    return load(
        prefix=prefix,
        dataset=dataset,
        subject=subject,
        template=template,
        datatype=datatype,
        derivative=mri_constants.Derivative.BrainMask,
    )


def get_masked_t1(t1, mask, smooth_kernel, normalize):
    if smooth_kernel == 0:
        smooth_kernel = None
    masked = nilearn.masking.apply_mask(
        imgs=t1, mask_img=mask, smoothing_fwhm=smooth_kernel
    )
    if normalize:
        masked = normalize_ndarray(masked)

    return masked


def get_masked_t1_curr(margs):
    t1, mask, args = margs
    smooth_kernel = args.smooth_kernel
    normalize = args.normalize
    return get_masked_t1(t1, mask, smooth_kernel, normalize)


def get_masked_t1s(args, t1s, supermask):
    results = []
    with Pool(args.cpus) as p:
        nt1s = len(t1s)
        map_args = zip(t1s, [supermask] * nt1s, [args] * nt1s)

        with tqdm.tqdm(desc="Masking reference", unit="image", total=nt1s) as pbar:
            for image in p.imap(get_masked_t1_curr, map_args):
                pbar.update()
                results.append(image)

    return np.stack(results), supermask


def get_template(template):
    image_path = tflow.get(
        template, desc=None, resolution=1, suffix="T1w", extension="nii.gz"
    )
    mask_path = tflow.get(
        template, desc="brain", resolution=1, suffix="mask", extension="nii.gz"
    )
    image = nibabel.load(image_path)
    mask = nibabel.load(mask_path)
    return mask_image(image, mask.get_fdata())
