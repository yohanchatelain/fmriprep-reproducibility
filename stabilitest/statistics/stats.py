import nilearn
import numpy as np
import significantdigits
from significantdigits import Error, Method

import stabilitest.MRI.mri_image as mri_image


def get_inputs(args):
    reference_t1s, reference_masks = mri_image.load(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type,
    )

    reference_sample_size = len(reference_t1s)

    print(f"Sample size: {reference_sample_size}")

    t1s_masked, supermask = mri_image.get_masked_t1s(
        args, reference_t1s, reference_masks
    )

    return t1s_masked, supermask


def compute_stats(args):
    t1s_masked, supermask = get_inputs(args)

    mean = np.mean(t1s_masked, axis=0)
    std = np.std(t1s_masked, axis=0)

    sig = significantdigits.significant_digits(
        array=t1s_masked,
        reference=mean,
        base=2,
        axis=0,
        error=Error.Relative,
        method=Method.CNH,
    )

    filename = "_".join(
        [
            args.reference_prefix,
            args.reference_dataset,
            args.reference_subject,
            args.reference_template,
            args.mask_combination,
            str(int(args.smooth_kernel)),
        ]
    )

    def save(x, name):
        print(f"Unmask {filename}")
        x_img = nilearn.masking.unmask(x, supermask)
        print(f"Save NumPy {name}")
        np.save(f"{filename}_{name}", x)
        print(f"Save Niffi {name}")
        x_img.to_filename(f"{filename}_{name}.nii")

    save(mean, "mean")
    save(std, "std")
    save(sig, "sig")
