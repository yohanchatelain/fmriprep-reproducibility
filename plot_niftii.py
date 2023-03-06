import nilearn.masking
import nilearn.plotting
import nibabel
import argparse
import numpy as np

symmetric_rules = ['shift', 'cut']


def shift_image(img):
    return img + abs(img.min())


def cut_image(img):
    img[img < 0] = 0
    return img


def log_image(img):
    img = np.log(img)
    return img

def load_img(args):
    img = nibabel.load(args.filename)
    if args.mask:
        mask = load_mask(args)
        img = nilearn.masking.apply_mask(img, mask)

    if args.force_symmetric == 'shift':
        img = shift_image(img)
    elif args.force_symmetric == 'cut':
        img = cut_image(img)

    if args.log:
        img = log_image(img)
        
    if args.mask:
        img = nilearn.masking.unmask(img, mask)

    return img


def load_mask(args):
    mask = nibabel.load(args.mask)
    return mask


def plot(args):
    img = load_img(args)
    if args.mask:
        mask = load_mask(args)
        img = nilearn.masking.apply_mask(img, mask)
        img = nilearn.masking.unmask(img, mask)

    cut_coords = tuple(map(int, args.cut_coords))
    view = nilearn.plotting.view_img(img,
                                     cut_coords=cut_coords,
                                     black_bg=True,
                                     bg_img=None,
                                     cmap=args.cmap,
                                     vmin=args.vmin,
                                     vmax=args.vmax,
                                     symmetric_cmap=False,
                                     title=args.title)

    view.open_in_browser()
    if args.output:
        nilearn.plotting.plot_img(img,
                                  cut_coords=cut_coords,
                                  black_bg=True,
                                  bg_img=None,
                                  output_file=args.output,
                                  cmap=args.cmap,
                                  threshold=args.threshold,
                                  colorbar=True,
                                  vmin=args.vmin,
                                  vmax=args.vmax,
                                  title=args.title)


def parse_args():
    parser = argparse.ArgumentParser('plot nifti image')
    parser.add_argument('--filename', required=True, help='filename of image')
    parser.add_argument('--mask',  help='filename of mask')
    parser.add_argument('--cmap', default='Greys_r',
                        help='cmap')
    parser.add_argument('--force-symmetric', choices=symmetric_rules,
                        help='Force the colorscale to by symmetric')
    parser.add_argument('--title', default='', help='Title')
    parser.add_argument('--cut-coords', default=(0, 0, 0),
                        nargs='+', help='cut-coords')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('--output', help='output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot(args)


if __name__ == '__main__':
    main()
