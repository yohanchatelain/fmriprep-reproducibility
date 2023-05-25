import numpy as np
import os
import re
import mri_printer
import mri_image
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import json
from filelock import FileLock

gmm_json_filename = "gmm_path.json"
regexp_id = re.compile(r"fmriprep_ds\d+_\d+\.\d+")


def get_repetition_id(filename):
    """
    fmriprep_<dataset>_<pid>.<id>
    """
    [path_id] = regexp_id.findall(filename)
    return path_id.split(".")[-1]


def get_key(filenames):
    ids = sorted([int(get_repetition_id(filename)) for filename in filenames])
    key = sum((1 << _id for _id in ids))
    return key


def get_gmm_path(gmm_dir, prefix, dataset, subject, template, mask, fwh, key):
    gmm_dir += os.path.sep
    prefix = prefix.replace(os.path.sep, "-")
    return (
        "_".join([gmm_dir, prefix, dataset, subject, template, mask, fwh, str(key)])
        + ".npy"
    )


def load_gmm(args, filenames):
    gmm_dir = args.gmm_cache
    prefix = args.reference_prefix
    subject = args.reference_subject
    dataset = args.reference_dataset
    template = args.reference_template
    mask = args.mask_combination
    fwh = str(args.smooth_kernel)
    key = str(get_key(filenames))
    ds = f"{dataset}_{subject}"

    gmm_json_path = os.path.join(gmm_dir, gmm_json_filename)

    with open(gmm_json_path, "r") as fi:
        gmm_json = json.load(fi)

    return gmm_json[ds][key][mask][fwh][template]


def dump_gmm(args, m, filenames):
    """
    gmm_path.json = {
        <dataset>_<subject> : {
            <key> : {
                repetitions_id = [],
                <mask> : {
                    <fwh> : {
                        <template> : <gmm_path>
                    }
                }
            }
        }
    }
    """
    gmm_dir = args.gmm_cache
    prefix = args.reference_prefix
    subject = args.reference_subject
    dataset = args.reference_dataset
    template = args.reference_template
    mask = args.mask_combination
    fwh = str(args.smooth_kernel)
    key = get_key(filenames)
    ds = f"{dataset}_{subject}"

    gmm_json_path = os.path.join(gmm_dir, gmm_json_filename)
    if not os.path.exists(gmm_json_path):
        gmm_json = {}
    else:
        with open(gmm_json_path, "r") as fi:
            gmm_json = json.load(fi)

    ds_dict = gmm_json.get(ds, {})
    key_dict = ds_dict.get(key, {"repetitions_id": filenames})
    mask_dict = key_dict.get(mask, {})
    fwh_dict = mask_dict.get(fwh, {})

    gmm_path = get_gmm_path(gmm_dir, prefix, dataset, subject, template, mask, fwh, key)

    fwh_dict[template] = gmm_path
    mask_dict[fwh] = fwh_dict
    key_dict[mask] = mask_dict
    ds_dict[key] = key_dict
    gmm_json[ds] = ds_dict

    with open(gmm_json_path, "w") as fo:
        json.dump(gmm_json, fo)

    np.save(file=gmm_path, arr=m, allow_pickle=True)


def get_gmm(reg_covar=10**-6):
    return GaussianMixture(
        n_components=2, covariance_type="diag", verbose=2, reg_covar=reg_covar
    )


def gmm_fit(x):
    reg_covar = 10**-6
    while reg_covar < 1:
        gmm = get_gmm(reg_covar)
        try:
            return gmm.fit(x), reg_covar
        except:
            reg_covar *= 10


def compute_gmm(args):
    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.reference_template,
        data_type=args.data_type,
    )

    kfold = KFold(len(reference_t1s))

    # leave-one-out
    # one_out
    # ones_in
    kfold_split = kfold.split(reference_t1s)

    lock = FileLock("mri_gmm.lock")

    for i, (ones_in, _) in enumerate(kfold_split, start=1):
        mri_printer.print_sep2(f"Round {i:2}")

        t1s_ref = reference_t1s[ones_in]
        masks_ref = reference_masks[ones_in]
        filenames = [img.get_filename() for img in t1s_ref]

        t1s_masked, _ = mri_image.mask_t1(args, t1s_ref, masks_ref)

        m, reg_covar = gmm_fit(t1s_masked)

        if args.verbose:
            print(f"fit gmm with T1 (reg_covar={reg_covar:.1e})")

        with lock:
            dump_gmm(args, m, filenames)


def main(args):
    cache = args.gmm_cache
    if not os.path.isdir(cache):
        os.makedirs(cache)

    compute_gmm(args)
