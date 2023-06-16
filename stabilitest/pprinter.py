import numpy as np
from enum import Enum


class BColors(Enum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


sep_h1 = "=" * 10
sep_h2 = "*" * 10
sep_h3 = "_" * 50

__verbose = False


def enable_verbose_mode():
    global __verbose
    __verbose = True


def verbose():
    return __verbose


def __pass_color_msg(msg):
    return f"{BColors.BOLD}{BColors.OKGREEN}{msg}{BColors.ENDC}"


def __fail_color_msg(msg):
    return f"{BColors.BOLD}{BColors.FAIL}{msg}{BColors.ENDC}"


def __test_color(passed):
    return __pass_color_msg if passed else __fail_color_msg


def __name_msg(name):
    return f"{BColors.BOLD}{name:<9}{BColors.ENDC} " if name else ""


def __ratio_msg(alpha, reject, tests, passed):
    ratio = reject / tests
    nb_digits = int(np.ceil(np.log10(tests)))
    reject_formatted = f"{reject:>{nb_digits}}"
    ratio_msg = f"[α={alpha:.4f}|{ratio*100:>6.3f}%|{reject_formatted}/{tests}]"
    return __test_color(passed)(ratio_msg)


def print_result(target, reject, tests, alpha, passed, name=None):
    name_msg = __name_msg(name)
    label_msg = passed
    ratio_msg = __ratio_msg(alpha, reject, tests, passed)
    filename_msg = target.get_filename()

    print(f"{name_msg} {label_msg} {ratio_msg} {filename_msg}")


def print_info(score, nsample, target, i=None, verbose=False):
    if verbose:
        name = f"{score} ({nsample} repetitions)"
        print_sep1(f"{name:^40}")
        header = f"Target ({i}): {target}"
        print_sep2(f"{header:^40}")


def print_name_method(name):
    print(f"{BColors.BOLD}{name}{BColors.ENDC}")


def print_debug(msg, verbose=False):
    if verbose:
        print(msg)


def print_sep(msg, sep):
    print(f"{sep} {msg} {sep}")


def print_sep1(msg):
    print_sep(msg, sep_h1)


def print_sep2(msg):
    print_sep(msg, sep_h2)


def print_sep3(msg):
    print_sep(msg, sep_h3)
