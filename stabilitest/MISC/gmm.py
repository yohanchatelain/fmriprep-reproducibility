import argparse
import os
import pickle

import nibabel
import numpy as np
import tqdm
from kneebow.rotor import Rotor
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def gmm(components, X):
    X = X.reshape((-1, 1))

    if components is not None:
        return GaussianMixture(n_components=components).fit(X)

    bic = []
    sil = []
    for n in range(1, 10):
        for _ in range(3):
            gm = GaussianMixture(n_components=n).fit(X)
            labels = gm.predict(X)
            _bic = np.mean([gm.bic(X) for _ in range(3)])
            _sil = np.mean(
                [
                    silhouette_score(X, labels) if len(set(labels)) > 1 else 0
                    for _ in range(3)
                ]
            )
        bic.append(_bic)
        sil.append(_sil)

    rotor = Rotor()
    z = np.dstack((np.arange(1, 10), np.gradient(bic))).reshape((-1, 2))
    rotor.fit_rotate(z)
    n1 = np.argmin(bic) + 1
    n2 = np.argmax(sil) + 1
    print(sil)
    n3 = rotor.get_elbow_index()
    print(n1, n2, n3)
    n = max(int(np.mean([n1, n2])), 1)
    print(n)

    return GaussianMixture(n_components=n).fit(X)


def get_image(args, i):
    components = [
        f"fmriprep_{args.dataset}_{i}",
        "fmriprep",
        args.subject,
        "anat",
        f"{args.subject}*preproc_T1w.nii.gz",
    ]
    return os.path.join(components)


def get_mask(args, i):
    components = [
        f"fmriprep_{args.dataset}_{i}",
        "fmriprep",
        args.subject,
        "anat",
        f"{args.subject}*brain_mask.nii.gz",
    ]
    return os.path.join(components)


def get_data(args):
    wildcard = "*"
    image_paths = get_image(args, wildcard)
    data = []
    supermask = None
    for image_path in image_paths:
        i = image_path.split(os.path.sep)[0].split("_")[-1]
        mask_path = get_mask(args, i)
        image = nibabel.load(image_path)
        mask = nibabel.load(mask_path)
        if supermask is None:
            supermask = mask.get_fdata()
        else:
            supermask = np.logical_and(mask.get_fdata(), supermask)
        masked_image = np.where(mask.get_fdata(), image.get_fdata(), 0)
        data.append(masked_image)
    return data, supermask


def parse_args():
    parser = argparse.ArgumentParser("Gaussian Mixture Model")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--components", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data, supermask = get_data(args)
    indices = np.array(np.nonzero(supermask)).T
    empty = np.full(data.shape[1:], 0)

    for i in range(1, 10):
        model = [gmm(i, data[(...,) + tuple(index)]) for index in tqdm.tqdm(indices)]
        for i, index in tqdm.tqdm(enumerate(indices)):
            empty[tuple(index)] = model[i]
        components = [args.dataset, args.subject, "AI", i]
        filename = "_".join(components)
        with open(filename, "wb") as fo:
            pickle.dump(empty, fo)


if "__main__" == __name__:
    main()
