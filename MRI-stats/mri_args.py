import argparse

NaN = float('nan')

default_templates = ['MNI152NLin2009cAsym', 'MNI152NLin6Asym']


def init_global_args(parser):
    parser.add_argument("--confidence", action="store",
                        default=0.95, type=float, help="Confidence", nargs='+')

    parser.add_argument("--reference-template", action="store",
                        required=True, help="Reference template")
    parser.add_argument("--data-type", action="store",
                        default='anat', choices=['anat'],
                        help="Data type")

    parser.add_argument("--reference-prefix", action='store',
                        required=True, help='Reference prefix path')
    parser.add_argument("--reference-dataset", action="store",
                        required=True, help="Dataset reference")
    parser.add_argument("--reference-subject", action="store",
                        required=True, help="Subject reference")

    parser.add_argument('--mask-non-normal-voxels', action='store_true',
                        help='Mask voxels that do not pass the normality test (Shapiro-Wild)')

    parser.add_argument('--k-fold-rounds', action='store',
                        type=int, default=5,
                        help='Number of K-fold rounds to perform')

    parser.add_argument('--smooth-kernel', '--fwh', '--fwhm', action='store',
                        type=float, default=0.0,
                        help='Size of the kernel smoothing')
    parser.add_argument('--mask-combination', action='store', type=str,
                        choices=['union', 'intersection', 'map'],
                        default='union',
                        help='Method to combine brain mask (map applies each brain mask to the image repetition)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the T1w to have [0,1] intensities')
    parser.add_argument('--output', action='store',
                        default='output.pkl', help='Output filename')
    parser.add_argument('--verbose', action='store_true',
                        help='verbose mode')
    parser.add_argument(
        '--score', choices=['z-score', 't-score'], default='z-score', help='Score to use')
    parser.add_argument('--gmm', action='store_true', help="Use GMM model")
    parser.add_argument('--compare-significant-digits',
                        action='store_true', help='Compare significant digits maps')
    parser.add_argument(
        '--gaussian-type', choices=['normal', 'skew', 'general'],
        default='normal',
        help='Gaussian normal distribution type')
    parser.add_argument('--parallel-fitting',
                        action='store_true', help='Parallel fitting')
    parser.add_argument('--cpus', default=1, type=int, help='Number of CPUs for multiprocessing')


def init_module_all_include(parser):
    msg = """
    Sanity check that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations, including the one being tested.
    """
    subparser = parser.add_parser("all-include", help=msg)
    init_global_args(subparser)
    subparser.add_argument('--gmm-paths', action='store',
                           help='Paths containing GMM objects')
    subparser.add_argument('--gmm-component', action='store', type=int,
                           help='Number of GMM components')


def init_module_all_exclude(parser):
    msg = """
    Sanity check that tests that the reference interval computed contains each
    reference observation. The reference interval is computed by using the all
    observations, excluding the one being tested.
    """
    subparser = parser.add_parser("all-exclude", help=msg)
    subparser.add_argument('--gmm-cache', default='.mri_cache',
                           help='Directory to cache gmm models')
    init_global_args(subparser)


def init_module_one(parser):
    msg = """
    Test that the target image is included into the statistical
    interval computed from the reference sample.
    """
    subparser = parser.add_parser("one", help=msg)
    init_global_args(subparser)
    subparser.add_argument("--target-prefix", action="store",
                           required=True, help="Target prefix path")
    subparser.add_argument("--target-dataset", action="store",
                           required=True, help="Dataset target")
    subparser.add_argument("--target-subject", action="store",
                           required=True, help="Subject target")
    subparser.add_argument("--target-template", action="store",
                           required=True, help="Target template")


def init_module_normality(parser):
    msg = """
    Apply a voxel-wise normality test
    """
    subparser = parser.add_parser("normality", help=msg)
    init_global_args(subparser)


def init_module_k_fold(parser):
    msg = """
    Sanity check that tests that the reference interval (train set)
    computed contain reference observations (test set).
    The train/test is splitted with a 80/20 ratio and
    is done K times.
    """
    subparser = parser.add_parser("k-fold", help=msg)
    init_global_args(subparser)


def init_module_stats(parser):
    msg = """
    Submodule for basics statistics (mean, std, sig)
    """
    subparser = parser.add_parser("stats", help=msg)
    init_global_args(subparser)


def init_module_gmm(parser):
    msg = """
    Submodule for fitting Gaussian Mixture Model
    """
    subparser = parser.add_parser("gmm", help=msg)
    init_global_args(subparser)
    subparser.add_argument('--gmm-cache', default='.mri_cache',
                           help='Directory to cache gmm models')


def init_module_distance(parser):
    msg = """
    Submodule for computing various distances
    """
    subparser = parser.add_parser("distance", help=msg)
    init_global_args(subparser)


def init_module_snr(parser):
    msg = """
    Submodule for computing SNR
    """
    subparser = parser.add_parser("snr", help=msg)
    init_global_args(subparser)


def parse_args():
    parser = argparse.ArgumentParser(description='mri-stats', prog='mri-stats')
    subparser = parser.add_subparsers(title='MRI-stats submodules',
                                      help='MRI-stats submodules',
                                      dest='mri_test')
    init_module_all_exclude(subparser)
    init_module_all_include(subparser)
    init_module_one(subparser)
    init_module_normality(subparser)
    init_module_k_fold(subparser)
    init_module_stats(subparser)
    init_module_gmm(subparser)
    init_module_distance(subparser)
    init_module_snr(subparser)

    args, _ = parser.parse_known_args()

    return parser, args
