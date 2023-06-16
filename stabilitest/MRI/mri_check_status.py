import pickle
import numpy as np
import argparse
from collections import Counter

status_choices = {"success": True, "fail": False}

raw_stats = list()
fvr_stats = dict()
status_stats = dict()


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header1(confidence, dataset, subject, method, mean, std):
    msg = (
        f"{bcolors.BOLD}{bcolors.WARNING}Confidence: {confidence}{bcolors.ENDC}, "
        f"Dataset: {dataset}, "
        f"Subject: {subject}, "
        f"{bcolors.BOLD}{bcolors.OKBLUE}Method: {method}{bcolors.ENDC}, "
        f"Mean: {mean:.4f}, "
        f"Std: {std:.4f}"
    )
    print(f"{bcolors.BOLD}{msg}{bcolors.ENDC}")


def print_header2(status):
    _total = sum(status.values())
    msg = (
        f"Sample size: {_total} | "
        f"Success: {status[True]} | "
        f"Fail: {status[False]}"
    )
    print(f"{bcolors.BOLD}{msg}{bcolors.ENDC}")


def print_pass():
    print(f"{bcolors.BOLD}{bcolors.OKGREEN}Pass{bcolors.ENDC}")


def print_fail():
    print(f"{bcolors.BOLD}{bcolors.FAIL}Fail{bcolors.ENDC}")


def print_status(confidence, dataset, subject, method, status, to_pass, mean, std):
    _status = status_choices[to_pass]
    print_header1(confidence, dataset, subject, method, mean, std)
    print_header2(status)
    for row in raw_stats:
        _confidence = row["confidence"]
        _dataset = row["dataset"]
        _subject = row["subject"]
        _method = row["method"]
        _fvr = row["fvr"]
        _target = row["target"]
        if (
            _confidence == confidence
            and _dataset == dataset
            and _subject == subject
            and method == _method
        ):
            if _status:
                if 1 - _confidence < _fvr:
                    print(f"- [{_fvr:.4f}] {_target}")
            else:
                if 1 - confidence > _fvr:
                    print(f"- [{_fvr:.4f}] {_target}")
    if to_pass == "success":
        if status[False] == 0:
            print_pass()
        else:
            print_fail()
    if to_pass == "fail":
        if status[True] == 0:
            print_pass()
        else:
            print_fail()


def parse_row(row):
    raw_stats.append(row)
    confidence = row["confidence"]
    dataset = row["dataset"]
    subject = row["subject"]
    method = row["method"]
    fvr = row["fvr"]
    key = (confidence, dataset, subject, method)
    status = (1 - confidence) < fvr
    _x = fvr_stats.get(key, [])
    _x.append(fvr)
    fvr_stats[key] = _x
    _y = status_stats.get(key, [])
    _y.append(status)
    status_stats[key] = _y


def parse_file(filename):
    with open(filename, "rb") as fi:
        data = pickle.load(fi)
        for d in data:
            parse_row(d)


def compute_stats(status_check):
    for key, fvrs in fvr_stats.items():
        (confidence, dataset, subject, method) = key
        fvr_mean = np.mean(fvrs, dtype=np.float64)
        fvr_std = np.std(fvrs, dtype=np.float64)
        status = [(1 - confidence) >= fvr for fvr in fvrs]
        counters = Counter(status)
        print_status(
            confidence,
            dataset,
            subject,
            method,
            counters,
            status_check,
            fvr_mean,
            fvr_std,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="check status")
    parser.add_argument(
        "--status", required=True, choices=status_choices.keys(), help="Expected status"
    )
    parser.add_argument(
        "--filename", required=True, help="Filename that contains statistics"
    )

    args = parser.parse_args()
    return args


if "__main__" == __name__:
    args = parse_args()
    parse_file(args.filename)
    compute_stats(args.status)
