import inspect
import json
import os
from enum import Enum

import niworkflows.data
import bids


def get_nipreps_specs():
    """
    Retriev BIDS specs used by nipreps
    """
    path = os.path.dirname(inspect.getfile(niworkflows.data))
    specs_filename = os.path.join(path, "nipreps.json")
    with open(specs_filename, "r", encoding="utf-8") as fi:
        specs = json.load(fi)
    return specs["default_path_patterns"]


_nipreps_bids_pattern = get_nipreps_specs()


class Derivative(Enum):
    T1wPreproc = "t1w-preproc"
    BrainMask = "brain-mask"
    WMProbabilisticSegmentation = "wm-probseg"
    GMProbabilisticSegmentation = "gm-probseg"
    CSFProbabilisticSegmentation = "csf-probseg"


t1w_suffix = "T1w"
probseg_suffix = "probseg"
mask_suffix = "mask"
dseg_suffix = "dseg"

nifti_extension = ".nii"
nifti_gzip_extension = ".nii.gz"
json_extension = ".json"

wm_label = "WM"
gm_label = "GM"
csf_label = "CSF"

preproc_desc = "preproc"
brain_desc = "brain"


def get_nifti_extension(gzip=True):
    if gzip:
        return {"extension": nifti_gzip_extension}
    else:
        return {"extension": nifti_extension}


def get_t1w_preproc_entity(gzip=True):
    return {
        "suffix": t1w_suffix,
        "desc": preproc_desc,
    } | get_nifti_extension(gzip=gzip)


def get_brain_mask_entity(gzip=True):
    return {
        "suffix": mask_suffix,
        "desc": brain_desc,
    } | get_nifti_extension(gzip=gzip)


def get_wm_label_entity(gzip=True):
    return {"label": wm_label, "suffix": probseg_suffix} | get_nifti_extension(
        gzip=gzip
    )


def get_gm_label_entity(gzip=True):
    return {"label": gm_label, "suffix": probseg_suffix} | get_nifti_extension(
        gzip=gzip
    )


def get_csf_label_entity(gzip=True):
    return {"label": csf_label, "suffix": probseg_suffix} | get_nifti_extension(
        gzip=gzip
    )


def get_derivatives_path(dataset, subject, template, datatype, derivative):
    entities = {
        "dataset": dataset,
        "subject": subject,
        "space": template,
        "datatype": datatype,
    }
    if derivative == Derivative.T1wPreproc:
        entities |= get_t1w_preproc_entity()
    if derivative == Derivative.BrainMask:
        entities |= get_brain_mask_entity()
    if derivative == Derivative.WMProbabilisticSegmentation:
        entities |= get_wm_label_entity()
    if derivative == Derivative.GMProbabilisticSegmentation:
        entities |= get_gm_label_entity()
    if derivative == Derivative.CSFProbabilisticSegmentation:
        entities |= get_csf_label_entity()

    return bids.layout.writing.build_path(entities, _nipreps_bids_pattern)
