import numpy as np
import itertools
import mri_image
import mri_constants
import scipy.spatial.distance as sdist
import tqdm
import os
import pandas as pd


def _hash(args, ext):
    keys = dict(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        fwhm=args.smooth_kernel,
        reference_ext=ext,
    )

    _hash = "_".join(
        map(lambda k: str(k).replace(os.path.sep, "-").replace(".", "_"), keys.values())
    )
    return _hash


def _has_memoization(args, ext):
    _file = _hash(args, ext) + ".npy"
    print(f"check memoized value {_file}")
    return os.path.exists(_file)


def _get_memoized(args, ext):
    print("loading memoized value")
    _file = _hash(args, ext) + ".npy"
    return np.load(_file, allow_pickle=True)


def _memoizes(args, ext, reference_masked):
    print("saving memoized value")
    _file = _hash(args, ext) + ".npy"
    np.save(_file, reference_masked)


def _get_reference(args, ext):
    if _has_memoization(args, ext):
        references_masked = _get_memoized(args, ext)
    else:
        references, reference_masks = mri_image.get_reference(
            prefix=args.reference_prefix,
            subject=args.reference_subject,
            dataset=args.reference_dataset,
            template=args.template,
            data_type=args.data_type,
            reference_ext=ext,
        )
        references_masked, _ = mri_image.mask_t1(args, references, reference_masks)
        _memoizes(args, ext, references_masked)

    return references_masked


def compute_snr(args):
    t1 = _get_reference(args, mri_constants.t1_preproc_extension)
    mean = np.mean(np.mean(t1, axis=0))
    std = np.sqrt(np.var(t1, axis=0).sum())
    print(mean / std)


def main(args):
    compute_snr(args)
