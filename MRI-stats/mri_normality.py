import numpy as np
import tqdm
import scipy
import nibabel
import nilearn

import mri_image
import mri_printer


def get_normality_mask(normality_mask_dir):
    if normality_mask_dir is None:
        return None

    image = nibabel.load(normality_mask_dir)
    return image.get_fdata().astype('bool')


def test_normality(images, mask):
    '''
    Compute a voxel-wise normality test for all images
    Returns a binary Nifti image with voxels rejecting the normality test set to True
    '''

    # Retrieve all indices inside the mask
    indices = np.array(np.nonzero(mask)).T
    empty = np.full(images[0].get_fdata().shape, 0)

    # Count the number of voxels
    nb_voxels = indices.shape[0]
    nb_non_normal = 0

    # voxels is 4D array made with the N MCA repetitions
    voxels = np.array([image.get_fdata() for image in images])
    # We test the normality using the Shapiro-Wilk test
    # scipy.stats.shapiro returns a pair (W,p-value)
    # Normality is rejected if p-value < 0.05.
    # It gives a 3D array of boolean with True if the voxels rejects the normality test
    non_normal_voxels = [scipy.stats.shapiro(
        voxels[(...,) + tuple(index)])[1] < 0.05 for index in tqdm.tqdm(indices)]
    # We count the number of non-normal voxels

    for i, index in tqdm.tqdm(enumerate(indices)):
        empty[tuple(index)] = 1 if non_normal_voxels[i] else 0
        nb_non_normal += 1 if non_normal_voxels[i] else 0

    ratio = nb_non_normal/nb_voxels

    print(f'Card(Voxels not normal) = {nb_non_normal}')
    print(f'Card(Voxels)            = {nb_voxels}')
    print(f'non-normal voxel ratio   = {ratio:.2e} [{ratio*100:f}%]')

    return nibabel.Nifti1Image(empty, images[0].affine)


def run_test_normality(args):
    '''
    Run the non-normality test for the given inputs
    Save the non-normality brain Nifti image computed and
    returns the filename
    '''
    if not args.mask_non_normal_voxels:
        return None

    template = args.template
    dataset = args.reference_dataset
    subject = args.reference_subject

    reference, supermask = mri_image.get_reference(
        reference_prefix=args.reference_prefix,
        reference_subject=args.reference_subject,
        reference_dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type,
        mask_combination=args.mask_combination,
        normalize=args.normalize,
        smooth_kernel=args.smooth_kernel,
        normality_mask=None)

    mri_printer.print_sep1('Normality test')
    non_normal_image = test_normality(reference, supermask)

    filename = f'non-normal-{dataset}-{subject}-{template}.nii.gz'
    nibabel.save(non_normal_image, filename)

    return non_normal_image.get_filename()


def plot_normality_image(filename, template):
    template_img = mri_image.get_template(template)
    image = nibabel.load(filename)
    view = nilearn.plotting.view_img(image,
                                     cmap='Reds',
                                     cut_coords=(0, 0, 0),
                                     symmetric_cmap=False,
                                     opacity=0.5,
                                     black_bg=True,
                                     bg_img=template_img,
                                     title=f'Non-normal voxel {template}')
    return view
