import numpy as np
import tqdm
import scipy
import nibabel
import nilearn

import stabilitest.mri.image as image
import stabilitest.pprinter as pprinter


def get_normality_mask(normality_mask_dir):
    if normality_mask_dir is None:
        return None

    image = nibabel.load(normality_mask_dir)
    return image.get_fdata().astype("bool")


def test_normality(args, t1s, masks, alpha):
    """
    Compute a voxel-wise normality test for all images
    Returns a binary Nifti image with voxels rejecting the normality test set to True
    """

    t1_masked, supermask = image.get_masked_t1s(args, t1s, masks)
    (_, t1_shape) = t1_masked.shape

    nb_voxels = np.count_nonzero(supermask.get_fdata())

    # voxels is 4D array made with the N MCA repetitions
    # voxels = np.array([t1.get_fdata() for t1 in t1_masked])
    # We test the normality using the Shapiro-Wilk test
    # scipy.stats.shapiro returns a pair (W,p-value)
    # Normality is rejected if p-value < 0.05.
    # It gives a 3D array of boolean with True if the voxels rejects the
    # normality test
    shapiro_test = (
        scipy.stats.shapiro(t1_masked[..., index])[1] < alpha
        for index in tqdm.tqdm(range(t1_shape))
    )
    non_normal_voxels = np.fromiter(shapiro_test, bool)
    # # We count the number of non-normal voxels
    nb_non_normal = non_normal_voxels.sum()

    normality_image = nilearn.masking.unmask(non_normal_voxels, supermask)

    ratio = nb_non_normal / nb_voxels

    print(f"Card(Voxels not normal) = {nb_non_normal}")
    print(f"Card(Voxels)            = {nb_voxels}")
    print(f"non-normal voxel ratio   = {ratio:.2e} [{ratio*100:f}%]")

    return normality_image


def run_test_normality(args):
    """
    Run the non-normality test for the given inputs
    Save the non-normality brain Nifti image computed and
    returns the filename
    """

    template = args.reference_template
    dataset = args.reference_dataset
    subject = args.reference_subject
    mask_combination = args.mask_combination
    fwh = args.smooth_kernel

    filenames = []

    for confidence in args.confidence:
        alpha = 1 - confidence

        t1s, masks = image.load(
            prefix=args.reference_prefix,
            subject=args.reference_subject,
            dataset=args.reference_dataset,
            template=args.reference_template,
            data_type=args.data_type,
        )

        pprinter.print_sep1("Normality test")
        non_normal_image = test_normality(args, t1s, masks, alpha)

        fields = [
            "non-normal",
            dataset,
            subject,
            template,
            mask_combination,
            str(int(fwh)),
            f"{alpha:.3f}",
        ]

        filename = "-".join(fields) + ".nii.gz"
        nibabel.save(non_normal_image, filename)

        filenames.append(non_normal_image.get_filename())

    return filenames


def plot_normality_image(filename, template):
    template_img = image.get_template(template)
    image = nibabel.load(filename)
    view = nilearn.plotting.view_img(
        image,
        cmap="Reds",
        cut_coords=(0, 0, 0),
        symmetric_cmap=False,
        opacity=0.5,
        black_bg=True,
        bg_img=template_img,
        title=f"Non-normal voxel {template}",
    )
    return view
