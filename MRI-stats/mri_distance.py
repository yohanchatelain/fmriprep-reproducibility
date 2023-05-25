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


def compute_discrete_distance(df, info, array):
    pass

    bg = array == 0
    gm = array == 1
    wm = array == 2
    csf = array == 3

    tissues = {"bg": bg, "gm": gm, "wm": wm, "csf": csf}

    distances = ["dice", "hamming", "jaccard"]

    for distance in distances:
        print(distance)
        for name, tissue in tissues.items():
            print(name)
            pdist = sdist.pdist(tissue, distance)
            df.loc[-1] = info + [name, distance, pdist]
            df.index += 1


def compute_continuous_distance(df, info, array):
    distances = ["correlation", "euclidean", "jensenshannon", "sqeuclidean"]

    for distance in distances:
        print(distance)
        pdist = sdist.pdist(array, distance)
        df.loc[-1] = info + [distance, pdist]
        df.index += 1


def compute_distance(args):
    columns = [
        "class",
        "reference",
        "dataset",
        "subject",
        "template",
        "fwhm",
        "label",
        "metric",
        "distance",
    ]
    df = pd.DataFrame(columns=columns)
    dseg = _get_reference(args, mri_constants.dseg_extension)
    csf = _get_reference(args, mri_constants.csf_probseg_extension)
    gm = _get_reference(args, mri_constants.gm_probseg_extension)
    wm = _get_reference(args, mri_constants.wm_probseg_extension)

    info = [
        "discrete",
        args.reference_prefix,
        args.reference_dataset,
        args.reference_subject,
        args.template,
        args.smooth_kernel,
    ]
    compute_discrete_distance(df, info, dseg)

    info = [
        "continuous",
        args.reference_prefix,
        args.reference_dataset,
        args.reference_subject,
        args.template,
        args.smooth_kernel,
    ]
    compute_continuous_distance(df, info + ["csf"], csf)
    compute_continuous_distance(df, info + ["gm"], gm)
    compute_continuous_distance(df, info + ["wm"], wm)

    print(df)
    filename = _hash(args, "")
    df.to_csv(f"{filename}-table.csv")


def main(args):
    compute_distance(args)
