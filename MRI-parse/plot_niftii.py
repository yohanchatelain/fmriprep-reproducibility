import argparse

import matplotlib as mpl
import nibabel
import nilearn.masking
import nilearn.plotting
import numpy as np
import plotly.express as px

symmetric_rules = ["shift", "cut"]


def shift_image(img):
    data = img.get_fdata()
    data += abs(data.min())
    return nibabel.Nifti1Image(data, img.affine)


def cut_image(img):
    data = img.get_fdata()
    data[data < 0] = 0
    return nibabel.Nifti1Image(data, img.affine)


def log_image(img):
    data = img.get_fdata()
    log_data = np.log(data)
    return nibabel.Nifti1Image(log_data, img.affine)


def discretize(img):
    data = np.floor(img.get_fdata())
    return nibabel.Nifti1Image(data, img.affine)


def _load_img(filename):
    img = nibabel.load(filename)
    img = nibabel.Nifti1Image(img.get_fdata().astype(np.float32), img.affine)
    return img


def load_img(args):
    img = _load_img(args.filename)
    if args.mask:
        mask = load_mask(args)
        img = nilearn.masking.apply_mask(
            img, mask, smoothing_fwhm=args.fwhm if args.fwhm != 0 else None
        )
        img = nilearn.masking.unmask(img, mask)

    if args.log:
        img = log_image(img)

    if args.force_symmetric == "shift":
        img = shift_image(img)
    elif args.force_symmetric == "cut":
        img = cut_image(img)

    if args.discretize:
        img = discretize(img)

    return img


def load_mask(args):
    mask = nibabel.load(args.mask)
    return mask


def plot(args):
    img = load_img(args)
    if args.discretize:
        cmap = mpl.colormaps.get_cmap(args.cmap).resampled(
            np.abs(args.vmax - args.vmin)
        )
    else:
        cmap = mpl.colormaps.get_cmap(args.cmap)

    cut_coords = tuple(map(int, args.cut_coords))
    if args.show:
        view = nilearn.plotting.view_img(
            img,
            cut_coords=cut_coords,
            black_bg=True,
            bg_img=None,
            cmap=cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            symmetric_cmap=False,
            title=args.title,
            draw_cross=args.draw_cross,
        )

        view.open_in_browser()
    if args.output:
        nilearn.plotting.plot_img(
            img,
            cut_coords=cut_coords,
            black_bg=True,
            bg_img=None,
            output_file=args.output,
            cmap=cmap,
            threshold=args.threshold,
            vmin=args.vmin,
            vmax=args.vmax,
            title=args.title,
            colorbar=args.colorbar,
            draw_cross=args.draw_cross,
        )


def parse_args():
    parser = argparse.ArgumentParser("plot nifti image")
    parser.add_argument("--filename", required=True, help="filename of image")
    parser.add_argument("--mask", help="filename of mask")
    parser.add_argument("--cmap", default="Greys_r", help="cmap")
    parser.add_argument(
        "--force-symmetric",
        choices=symmetric_rules,
        help="Force the colorscale to by symmetric",
    )
    parser.add_argument("--title", default="", help="Title")
    parser.add_argument("--cut-coords", default=(0, 0, 0), nargs="+", help="cut-coords")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--output", help="output file")
    parser.add_argument("--colorbar", action="store_true", help="Show colorbar")
    parser.add_argument("--draw-cross", action="store_true", help="Draw cross")
    parser.add_argument(
        "--show", action="store_true", help="Dynamic visualization in browser"
    )
    parser.add_argument("--fwhm", type=float, help="Apply kernel smoothing (mm)")
    parser.add_argument("--discretize", action="store_true", help="Discretize values")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot(args)


if __name__ == "__main__":
    main()
