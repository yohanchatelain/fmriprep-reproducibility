import nilearn.masking
import nilearn.plotting
import nibabel
import argparse
import numpy as np

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


def load_img(args):
    img = nibabel.load(args.filename)
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

    return img


def load_mask(args):
    mask = nibabel.load(args.mask)
    return mask


def plot(args):
    img = load_img(args)

    cut_coords = tuple(map(int, args.cut_coords))
    if args.show:
        view = nilearn.plotting.view_img(
            img,
            cut_coords=cut_coords,
            black_bg=True,
            bg_img=None,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            symmetric_cmap=False,
            title=args.title,
        )

        view.open_in_browser()
    if args.output:
        nilearn.plotting.plot_img(
            img,
            cut_coords=cut_coords,
            black_bg=True,
            bg_img=None,
            output_file=args.output,
            cmap=args.cmap,
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
    parser.add_argument("--draw_cross", action="store_true", help="Draw cross")
    parser.add_argument(
        "--show", action="store_true", help="Dynamic visualization in browser"
    )
    parser.add_argument("--fwhm", type=float, help="Apply kernel smoothing (mm)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    plot(args)


if __name__ == "__main__":
    main()
