import numpy as np
import os
import re
import mri_printer
import mri_image
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import json

gmm_json_filename = 'gmm_path.json'
regexp_id = re.compile(r'fmriprep_ds\d+_\d+\.\d+')


def get_repetition_id(filename):
    '''
    fmriprep_<dataset>_<pid>.<id>
    '''
    [path_id] = regexp_id.findall(filename)
    return path_id.split('.')[-1]


def get_key(filenames):
    ids = sorted([int(get_repetition_id(filename)) for filename in filenames])
    key = sum((1 << _id for _id in ids))
    return key


def get_gmm_path(gmm_dir, prefix, dataset, subject, template, mask, fwh, key):
    gmm_dir += os.path.sep
    prefix = prefix.replace(os.path.sep, '-')
    return '_'.join([gmm_dir, prefix, dataset, subject, template, mask, fwh, key]) + '.npy'


def dump_gmm(args, m, filenames):
    '''
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
    '''
    gmm_dir = args.gmm_cache
    prefix = args.reference_prefix
    subject = args.reference_subject
    dataset = args.reference_dataset
    template = args.template
    mask = args.mask_combination
    fwh = str(args.smooth_kernel)
    key = get_key(filenames)
    ds = f'{dataset}_{subject}'

    gmm_json = None
    with open(gmm_json_filename, 'r') as fi:
        gmm_json = json.load(fi)

    ds_dict = gmm_json.get(ds, {})
    key_dict = ds_dict.get(key, {'repetitions_id': filenames})
    mask_dict = key_dict.get(mask, {})
    fwh_dict = mask_dict.get(fwh, {})

    gmm_path = get_gmm_path(gmm_dir, prefix, dataset,
                            subject, template, mask, fwh, key)

    fwh_dict[template] = gmm_path
    mask_dict[fwh] = fwh_dict
    key_dict[mask] = mask_dict
    ds_dict[key] = key_dict
    gmm_json[ds] = ds_dict

    with open(gmm_json_filename, 'w') as fo:
        json.dump(gmm_json, fo)

    np.save(file=gmm_path, arr=m, allows_pickle=True)


def compute_gmm(args):

    reference_t1s, reference_masks = mri_image.get_reference(
        prefix=args.reference_prefix,
        subject=args.reference_subject,
        dataset=args.reference_dataset,
        template=args.template,
        data_type=args.data_type)

    kfold = KFold(len(reference_t1s))

    # leave-one-out
    # one_out
    # ones_in
    kfold_split = kfold.split(reference_t1s)

    gmm = GaussianMixture(n_components=2, covariance_type='diag')

    for i, (ones_in, _) in enumerate(kfold_split, start=1):
        mri_printer.print_sep2(f'Round {i:2}')

        t1s_ref = reference_t1s[ones_in]
        masks_ref = reference_masks[ones_in]
        filenames = [img.get_filename() for img in t1s_ref]

        t1s_masked, supermask = mri_image.mask_t1(
            t1s_ref, masks_ref,
            args.mask_combination,
            args.smooth_kernel)

        m = gmm.fit(t1s_masked)

        dump_gmm(args, m, filenames)


def main(args):
    cache = args.gmm_path
    if not os.path.isdir(cache):
        os.makedirs(cache)

    compute_gmm(args)
