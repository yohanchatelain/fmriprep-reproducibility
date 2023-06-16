from stabilitest.sample import Sample
from stabilitest.MRI import mri_image
import numpy as np


class MRISample(Sample):
    def __init__(self, args):
        self.args = args
        self.t1ws = None
        self.t1ws_metadata = None
        self.brain_masks = None
        self.brain_masks_metadata = None
        self.supermask = None
        # T1ws masked, smoothed and min-max scaled
        self.preproc_t1ws = None

    def get_size(self):
        return self.t1ws.shape[0]

    def get_observation_shape(self):
        return self.t1ws.shape[1:]

    def _get_metadata(self, nifti):
        _metadata = [dict(n.header) | {"filename": nifti.get_filename()} for n in nifti]
        return np.array(_metadata)

    def load_t1w(self, prefix, dataset, subject, template, datatype):
        self.t1ws = mri_image.load_t1w(
            prefix=prefix,
            dataset=dataset,
            subject=subject,
            template=template,
            datatype=datatype,
        )
        self.t1ws_metadata = self._get_metadata(self.t1ws)

    def load_brain_mask(self, prefix, dataset, subject, template, datatype):
        self.brain_masks = mri_image.load_brain_mask(
            prefix=prefix,
            dataset=dataset,
            subject=subject,
            template=template,
            datatype=datatype,
        )
        self.brain_masks_metadata = self._get_metadata(self.t1ws)

    def load_t1_and_brain_maks(self, prefix, dataset, subject, template, datatype):
        self.load_t1w(prefix, dataset, subject, template, datatype)
        self.brain_masks(prefix, dataset, subject, template, datatype)

    def __compute_supermask(self, force=False):
        if force or self.supermask is None:
            self.supermask = mri_image.combine_mask(
                self.brain_masks, self.args.mask_combination
            )

    def __mask_sample(self, force=False):
        if self.preproc_t1ws is None or force:
            self.__compute_supermask(force)
            self.preproc_t1ws = mri_image.get_masked_t1s(
                self.args, self.t1ws, self.supermask
            )

    def __parse_index(self, indexes):
        if indexes is None:
            return ...
        if isinstance(indexes, int):
            return np.array(indexes)
        if isinstance(indexes, list):
            return np.array(indexes)

    def get_subsample(self, indexes=None, force=False):
        self.__mask_sample(force)
        return self.preproc_t1ws[self.__parse_index(indexes)]

    def get_filenames(self, indexes=None):
        meta = self.t1ws_metadata[self.__parse_index(indexes)]
        return [m["filename"] for m in meta]

    def resample(self, target):
        self.preproc_t1ws = mri_image.resample_image(self.preproc_t1ws, target)
