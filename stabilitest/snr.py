import numpy as np

import stabilitest.mri.constants as mri_constants
import stabilitest.mri.image as mri_image


def _get_reference(args, ext):
    references, reference_masks = mri_image.load(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        extension=ext,
    )
    references_masked, _ = mri_image.get_masked_t1s(args, references, reference_masks)
    return references_masked


def compute_snr(args, x):
    mean = np.mean(np.mean(x, axis=0))
    std = np.sqrt(np.var(x, axis=0).sum())
    return mean / std


def main(args):
    t1 = _get_reference(args, mri_constants.t1w_preproc_extension)
    compute_snr(args, t1)
