import nilearn.masking
import json
import nilearn.plotting
import os
import stabilitest.MRI.mri_image as mri_image
from PIL import Image
import argparse
import tempfile


def get_images(args):
    t1s, masks = mri_image.load(
        args.prefix,
        args.subject,
        args.dataset,
        args.template,
        args.data_type,
        args.normalize,
    )
    if args.mask_type in ("union", "intersection"):
        t1s_masked, supermask = mri_image.get_masked_t1s(args, t1s, masks)
    else:
        # Do not apply mask
        t1s_masked = nilearn.image.smooth_img(t1s, args.fwh)
        for t1, t2 in zip(t1s_masked, t1s):
            t1.set_filename(t2.get_filename())
        supermask = None
    return t1s_masked, supermask


def make_gif(args, t1s, supermask):
    prefix = [
        "",
        args.dataset,
        args.subject,
        args.mask_type,
        args.template,
        str(int(args.fwh)),
        args.extra,
    ]

    tmp = tempfile.mkdtemp(suffix="_".join(prefix), dir=".")

    if args.verbose:
        print(f"Temporary dir: {tmp}")

    fig_map = {}
    output_files = []
    for i, t1 in enumerate(t1s):
        if args.verbose:
            print(f"plotting {i}/{len(t1s)-1}")

        if supermask is not None:
            t1 = nilearn.masking.unmask(t1, supermask)

        fig_map[i] = t1.get_filename()
        print(t1.get_filename())
        filename = t1.get_filename().replace(os.path.sep, "_").replace(".", "_")
        output_file = os.path.join(tmp, filename + ".png")
        fig_map[i] = output_file
        output_files.append(output_file)
        nilearn.plotting.plot_anat(
            t1, cut_coords=(0, 0, 0), output_file=output_file, draw_cross=False, title=i
        )

    with open(os.path.join(tmp, "map.json"), "w") as fo:
        json.dump(fig_map, fo)

    gif_filename = os.path.join(tmp, "anat")

    if args.verbose:
        print(f"Making GIF {gif_filename}")

    frames = [Image.open(o) for o in output_files]
    frame_1 = frames[0]
    frame_1.save(
        gif_filename,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=100,
        loop=0,
    )


def parse_args():
    parser = argparse.ArgumentParser("GIF maker")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--template", default="MNI152NLin2009cAsym")
    parser.add_argument("--data-type", default="anat")
    parser.add_argument(
        "--mask-type", default="", choices=("union", "intersection", "raw")
    )
    parser.add_argument("--fwh", type=float, required=True)
    parser.add_argument("--extra", default="")
    parser.add_argument("--output", default="output.gif")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.verbose:
        print(args)
    t1s_masked, supermask = get_images(args)
    T1s, masks = mri_image.load(
        args.prefix,
        args.subject,
        args.dataset,
        args.template,
        args.data_type,
        args.normalize,
    )
    make_gif(args, T1s, None)


if "__main__" == __name__:
    main()
