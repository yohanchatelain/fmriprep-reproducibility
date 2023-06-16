import os
import argparse
import nilearn.image
import nilearn.masking
import nibabel
import numpy as np


def get_image(args):
    return nibabel.load(args.image)


def get_mask(args):
    mask = nibabel.load(args.mask) if args.mask else None
    return mask


def get_output(args, ratio):
    if args.output is None:
        output = os.path.basename(args.image)
    else:
        output = args.output

    output = f'{ratio}_{output}'
    return output


def apply_mask(img, mask):
    if mask:
        voxels = nilearn.masking.apply_mask(img, mask)
    else:
        voxels = img.get_fdata()
    return voxels


def unmask(voxels, img, mask):
    if mask:
        noised_img = nilearn.masking.unmask(voxels, mask)
    else:
        noised_img = nibabel.Nifti1Image(voxels, img.affine)
    return noised_img


def make_noised_image(args, rng):

    image = get_image(args)
    mask = get_mask(args)

    voxels = apply_mask(image, mask)

    nb_voxels = voxels.size
    indices, _ = zip(*list(np.ndenumerate(voxels)))
    indices = set(indices)

    if args.verbose:
        print('Make noised image')
        print(f'{nb_voxels} voxels')

    previous_ratio = 0
    for ratio in args.ratios:
        size = int((ratio / 100 - previous_ratio) * nb_voxels)
        previous_ratio = ratio / 100

        if args.verbose:
            print(f'Create noised image with {ratio}% of dead voxels')
            print(f'Adds {size} dead voxels to the image')

        noise_index = rng.choice(nb_voxels, size)
        indices -= set(noise_index)
        for index in noise_index:
            voxels[index] = 0

        noised_img = unmask(voxels, image, mask)
        output = get_output(args, ratio)
        nibabel.save(noised_img, filename=output)

    return noised_img


def parse_args():
    parser = argparse.ArgumentParser('dead-voxels')
    parser.add_argument('--ratios', required=True,
                        type=float, nargs='+', help='List of ratios')
    parser.add_argument(
        '--output', help='Output prefix filename. Use image name as default')
    parser.add_argument('--image', required=True, help='Image to noise')
    parser.add_argument('--mask', help='Mask to apply to the image')
    parser.add_argument('--seed', default=0, help='Set seed')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)
    make_noised_image(args, rng)


if '__main__' == __name__:
    main()
