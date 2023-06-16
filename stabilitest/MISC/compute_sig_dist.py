import argparse
import os
import stabilitest.MRI.mri_image as mri_image
from sklearn.model_selection import KFold
import significantdigits
import numpy as np
import pickle


def compute_sig(array, reference):
    sig = significantdigits.significant_digits(
        array, reference=reference, method=significantdigits.Method.General
    )
    return sig


def compute_k_fold_sig(args, reference_t1, reference_mask):
    kfold = KFold(len(reference_t1))

    t1_masked, _ = mri_image.get_masked_t1s(args, reference_t1, reference_mask)
    global_reference = np.mean(t1_masked, axis=0)
    global_sig = compute_sig(t1_masked, global_reference)

    hists = []

    for i, _ in kfold.split(reference_t1):
        t1_test = t1_masked[i]

        reference = np.mean(t1_test, axis=0)
        sig = compute_sig(t1_test, reference)

        diff = np.abs(global_sig - sig)
        hist = np.histogram(diff, bins=range(13))
        hists.append(hist)

    return hists


def get_filename(args):
    prefix = args.reference_prefix.replace(os.path.sep, "-")
    return "_".join(
        map(
            str,
            [
                prefix,
                args.reference_dataset,
                args.reference_subject,
                args.template,
                args.smooth_kernel,
            ],
        )
    )


def compute_sig_loo(args):
    """
    compute significant digits for leave-one-out
    """
    reference_t1s, reference_masks = mri_image.load(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
    )

    reference_sample_size = len(reference_t1s)

    print(f"Sample size: {reference_sample_size}")

    hists = compute_k_fold_sig(args, reference_t1s, reference_masks)

    filename = get_filename(args)

    with open(filename, "wb") as fo:
        pickle.dump(hists, fo)


default_templates = ["MNI152NLin2009cAsym", "MNI152NLin6Asym"]


def init_global_args(parser):
    parser.add_argument(
        "--template",
        action="store",
        choices=default_templates,
        required=True,
        help="Template",
    )
    parser.add_argument(
        "--data-type",
        action="store",
        default="anat",
        choices=["anat"],
        required=True,
        help="Data type",
    )
    parser.add_argument(
        "--reference-prefix",
        action="store",
        required=True,
        help="Reference prefix path",
    )
    parser.add_argument(
        "--reference-dataset", action="store", required=True, help="Dataset reference"
    )
    parser.add_argument(
        "--reference-subject", action="store", required=True, help="Subject reference"
    )
    parser.add_argument(
        "--smooth-kernel",
        "--fwh",
        "--fwhm",
        action="store",
        type=float,
        default=0.0,
        help="Size of the kernel smoothing",
    )
    parser.add_argument(
        "--mask-combination",
        action="store",
        type=str,
        choices=["union", "intersection", "map"],
        default="union",
        help="Method to combine brain mask (map applies each brain mask to the image repetition)",
    )


def parse_args():
    parser = argparse.ArgumentParser("sig diff")
    init_global_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    compute_sig_loo(args)


if __name__ == "__main__":
    main()
