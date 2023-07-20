import argparse
import csv
import json
from multiprocessing import Pool
import os
import pickle
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import joblib
import numpy as np
from icecream import ic

import stabilitest.main

stats_methods = [
    "pce",
    # "fdr_TSBY",
    # "fdr_TSBH",
    "fdr_BY",
    # "fdr_BH",
    # "fwe_simes_hochberg",
    # "fwe_holm_bonferroni",
    # "fwe_holm_sidak",
    "fwe_sidak",
    "fwe_bonferroni",
]


parameters_dict = {
    "loo": {
        "reference_architecture": ["narval-AMD"],
        "reference_version": ["20.2.1"],
        "reference_template": ["MNI152NLin2009cAsym"],
        "reference_perturbation": ["rr", "rs"],
        "target_architecture": [None],
        "target_version": [None],
        "target_template": [None],
        "target_perturbation": [None],
    },
    "ieee": {
        "reference_architecture": ["narval-AMD"],
        "reference_version": ["20.2.1"],
        "reference_template": ["MNI152NLin2009cAsym"],
        "reference_perturbation": ["rr", "rs"],
        "target_architecture": ["narval-AMD"],
        "target_version": ["20.2.1"],
        "target_template": ["MNI152NLin2009cAsym"],
        "target_perturbation": ["ieee"],
    },
    "template": {
        "reference_architecture": ["narval-AMD"],
        "reference_version": ["20.2.1"],
        "reference_template": ["MNI152NLin2009cAsym"],
        "reference_perturbation": ["rr", "rs"],
        "target_architecture": ["narval-AMD"],
        "target_version": ["20.2.1"],
        "target_prefix": ["template"],
        "target_perturbation": ["ieee"],
        "target_template": [
            "MNI152NLin2009cAsymNoised0001",
            "MNI152NLin2009cAsymNoised0005",
            "MNI152NLin2009cAsymNoised0010",
            "MNI152NLin2009cAsymNoised0050",
            "MNI152NLin2009cAsymNoised0100",
            "MNI152NLin2009cAsymNoised0200",
            "MNI152NLin2009cAsymNoised0300",
            "MNI152NLin2009cAsymNoised0400",
            "MNI152NLin2009cAsymNoised0500",
            "MNI152NLin2009cAsymNoised0600",
            "MNI152NLin2009cAsymNoised0700",
            "MNI152NLin2009cAsymNoised0800",
        ],
    },
    "version": {
        "reference_architecture": ["narval-AMD"],
        "reference_version": ["20.2.1"],
        "reference_template": ["MNI152NLin2009cAsym"],
        "reference_perturbation": ["rr", "rs"],
        "target_architecture": [None],
        "target_version": [
            "20.2.0",
            "20.2.1",
            "20.2.2",
            "20.2.3",
            "20.2.4",
            "20.2.5",
            "21.0.4",
            "22.1.1",
            "23.0.0",
        ],
        "target_template": [None],
        "target_perturbation": ["ieee"],
    },
    "architecture": {
        "reference_architecture": ["narval-AMD"],
        "reference_version": ["20.2.1"],
        "reference_template": ["MNI152NLin2009cAsym"],
        "reference_perturbation": ["rr", "rs"],
        "target_architecture": [
            "narval-AMD",
            "cedar-broadwell",
            "cedar-cascade",
            "cedar-skylake",
            "graham-broadwell",
            "graham-cascade",
        ],
        "target_version": ["20.2.1"],
        "target_template": [None],
        "target_perturbation": ["ieee"],
    },
}


def test_args(test):
    if test == "loo":
        return ["cross-validation", "--model=loo"]
    if test == "ieee":
        return ["test"]
    if test == "template":
        return ["test"]
    if test == "version":
        return ["test"]
    if test == "architecture":
        return ["test"]


def get_prefix_reference(args, test, architecture, version, template, perturbation):
    return os.path.join(args.output_path, version, perturbation)


def get_prefix_target(args, test, architecture, version, template, perturbation):
    if test == "loo":
        return get_prefix_reference(
            args, test, architecture, version, template, perturbation
        )
    if test == "ieee":
        return get_prefix_reference(
            args, test, architecture, version, template, perturbation
        )
    if test == "template":
        return os.path.join(args.output_path, version, "template", perturbation)
    if test == "version":
        return get_prefix_reference(
            args, test, architecture, version, template, perturbation
        )
    if test == "architecture":
        return os.path.join(
            args.output_path, version, "arch", architecture, perturbation
        )


def _make_parameters_matrix(args, test, inputs):
    param = parameters_dict[test]
    _iter_reference = product(
        param["reference_architecture"],
        param["reference_version"],
        param["reference_template"],
        param["reference_perturbation"],
    )
    path = test
    counter = 0
    config_matrix = []
    for pref in _iter_reference:
        _iter_target = product(
            param["target_architecture"],
            param["target_version"],
            param["target_template"],
            param["target_perturbation"],
        )
        for ptar in _iter_target:
            for (
                dataset_reference,
                subject_reference,
                dataset_target,
                subject_target,
            ) in inputs:
                for fwhm in range(args.fwhm + 1):
                    if args.configurations != 0 and args.configurations <= counter:
                        break
                    (
                        architecture_ref,
                        version_ref,
                        template_ref,
                        perturbation_ref,
                    ) = pref
                    (
                        architecture_tar,
                        version_tar,
                        template_tar,
                        perturbation_tar,
                    ) = tuple(tar if tar else ref for (tar, ref) in zip(ptar, pref))

                    prefix_reference = get_prefix_reference(
                        args,
                        test,
                        architecture_ref,
                        version_ref,
                        template_ref,
                        perturbation_ref,
                    )
                    prefix_target = get_prefix_target(
                        args,
                        test,
                        architecture_tar,
                        version_tar,
                        template_tar,
                        perturbation_tar,
                    )
                    output = os.path.join(path, "pickle", f"{counter}.pkl")
                    pickle_path = os.path.join(path, "pickle")
                    os.makedirs(pickle_path, exist_ok=True)
                    confidences = map(lambda c: f"{c:g}", args.confidence)
                    config = test_args(test)
                    config += (
                        ["--confidence"]
                        + list(confidences)
                        + ["--multiple-comparison-tests"]
                        + stats_methods
                        + [
                            f"--output={output}",
                            f"--cpus={args.cpus}",
                            "smri",
                            f"--reference-architecture={architecture_ref}",
                            f"--reference-perturbation={perturbation_ref}",
                            f"--reference-template={template_ref}",
                            f"--reference-version={version_ref}",
                            f"--reference-prefix={prefix_reference}",
                            f"--reference-dataset={dataset_reference}",
                            f"--reference-subject={subject_reference}",
                            f"--target-architecture={architecture_tar}",
                            f"--target-perturbation={perturbation_tar}",
                            f"--target-template={template_tar}",
                            f"--target-version={version_tar}",
                            f"--target-prefix={prefix_target}",
                            f"--target-dataset={dataset_target}",
                            f"--target-subject={subject_target}",
                            f"--fwhm={fwhm}",
                        ]
                    )
                    record = {
                        "model": test,
                        "reference_architecture": architecture_ref,
                        "reference_perturbation": perturbation_ref,
                        "reference_template": template_ref,
                        "reference_version": version_ref,
                        "reference_prefix": prefix_reference,
                        "reference_dataset": dataset_reference,
                        "reference_subject": subject_reference,
                        "target_architecture": architecture_tar,
                        "target_perturbation": perturbation_tar,
                        "target_template": template_tar,
                        "target_version": version_tar,
                        "target_prefix": prefix_target,
                        "target_dataset": dataset_target,
                        "target_subject": subject_target,
                        "fwhm": str(fwhm),
                        "pickle": output,
                    }
                    config_matrix.append((config, record))
                    counter += 1
    return config_matrix


def make_configuration_matrix_loo(args, inputs):
    test = "loo"
    inputs = [(d1, s1, d1, s1) for (d1, s1) in inputs]
    return _make_parameters_matrix(args, test, inputs)


def make_configuration_matrix_ieee(args, inputs):
    test = "ieee"
    inputs = [(d1, s1, d2, s2) for ((d1, s1), (d2, s2)) in product(inputs, inputs)]
    return _make_parameters_matrix(args, test, inputs)


def make_configuration_matrix_template(args, inputs):
    test = "template"
    inputs = [(d1, s1, d1, s1) for (d1, s1) in inputs]
    return _make_parameters_matrix(args, test, inputs)


def make_configuration_matrix_version(args, inputs):
    test = "version"
    inputs = [(d1, s1, d1, s1) for (d1, s1) in inputs]
    return _make_parameters_matrix(args, test, inputs)


def make_configuration_matrix_architecture(args, inputs):
    test = "architecture"
    inputs = [(d1, s1, d1, s1) for (d1, s1) in inputs]
    return _make_parameters_matrix(args, test, inputs)


confidence_default = list(np.linspace(0.5, 0.95, 10)) + [0.99, 0.995, 0.999]


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser("run-test")
    parser.add_argument("--output-path", default="outputs", help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Print tests to run")
    parser.add_argument("--cpus", default=1, help="Number of CPUs to use")
    parser.add_argument(
        "--confidence",
        type=float,
        default=confidence_default,
        nargs="+",
        help="Confidence value",
    )
    parser.add_argument("--test", choices=tests.keys(), help="Run specific test")
    parser.add_argument("--inputs", required=True, help="Dataset inputs")
    parser.add_argument(
        "--fwhm", type=int, default=20, help="Maximum FWHM value to test"
    )
    parser.add_argument(
        "--force", action="store_true", help="Run test even if output directories exist"
    )
    parser.add_argument(
        "--configurations",
        type=int,
        default=0,
        metavar="N",
        help="Run the N first configurations",
    )
    args = parser.parse_args()
    return args


def run_stabilitest(args):
    stabilitest.main(args)


def dry_run(parameters):
    for parameter in parameters:
        print(" ".join(parameter))


def make_configuration_matrix(args, test, inputs):
    if test == "loo":
        return make_configuration_matrix_loo(args, inputs)
    if test == "ieee":
        return make_configuration_matrix_ieee(args, inputs)
    if test == "template":
        return make_configuration_matrix_template(args, inputs)
    if test == "version":
        return make_configuration_matrix_version(args, inputs)
    if test == "architecture":
        return make_configuration_matrix_architecture(args, inputs)


def dump_table(args, table_filename, table):
    if args.dry_run:
        fieldnames = list(table[0].keys())
        writer = csv.writer(sys.stdout)
        print(",".join(fieldnames))
        for row in table:
            writer.writerow(row.values())
        return

    with open(table_filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = list(table[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)


def get_inputs(args):
    with open(args.inputs, "r", encoding="utf-8") as fi:
        ds = json.load(fi)
        return [
            (dataset, subject)
            for (dataset, subjects) in ds.items()
            for subject in subjects
        ]


def get_table(args, table_filename):
    table = []
    table_map = {}

    if args.force:
        return table, table_map

    if os.path.exists(table_filename):
        with open(table_filename, "r", encoding="utf-8") as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=",")
            for row in csvreader:
                table.append(row)
                table_map[joblib.hash(row)] = row
    else:
        table_map = {}
        table = []

    return table, table_map


def process_configuration(configuration):
    ic(configuration)
    stabilitest.main.main(configuration)


def skip_cached(configurations, table_map):
    to_process = []
    for configuration, record in configurations:
        if joblib.hash(record) in table_map:
            print(f"Skip cached {configuration}")
            continue
        to_process.append((configuration, record))
    return to_process


def run_test(args, test, inputs):
    path = test

    if args.force:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    # Table to map arguments to output pickle file
    table_filename = os.path.join(path, "table.csv")
    table, table_map = get_table(args, table_filename)
    inputs = get_inputs(args)
    configurations = make_configuration_matrix(args, test, inputs)

    configurations = skip_cached(configurations, table_map)

    configs, records = zip(*configurations)

    config_partition = np.array_split(configs, 10)
    records_partition = np.array_split(records, 10)

    n_jobs = 4

    for config_chunk, record_chunk in zip(config_partition, records_partition):
        if args.dry_run:
            print(config_chunk, sep="\n")
        else:
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(process_configuration)(c) for c in config_chunk
            )
            table.extend(record_chunk)
            dump_table(args, table_filename, table)

    # for (
    #     configuration,
    #     record,
    # ) in configurations:
    #     if joblib.hash(record) in table_map:
    #         print(f"Skip cached {configuration}")
    #         continue
    #     if args.dry_run:
    #         print(" ".join(configuration))
    #     else:
    #         config.append(process_configuration(configuration))

    #     table.append(record)

    #     dump_table(args, table_filename, table)
    # config.compute()


def test_loo(args, inputs):
    run_test(args, "loo", inputs)


def test_ieee(args, inputs):
    run_test(args, "ieee", inputs)


def test_template(args, inputs):
    run_test(args, "template", inputs)


def test_version(args, inputs):
    run_test(args, "version", inputs)


def test_architecture(args, inputs):
    run_test(args, "architecture", inputs)


tests = {
    "ieee": test_ieee,
    "template": test_template,
    "version": test_version,
    "architecture": test_architecture,
    "loo": test_loo,
}


def main():
    args = parse_args()
    inputs = get_inputs(args)
    if args.test:
        tests[args.test](args, inputs)
    else:
        for test in tests:
            tests[test](args, inputs)


if "__main__" == __name__:
    dask.config.set(pool=ThreadPoolExecutor(4))
    main()

# Workers                      1  | 10    | 20   | 40    | 80 |
# -------------------------------------------------------------
# loo               x 336  -> 22h | 2.2 h | 1.1h | 0.55h | 0.25h
# ieee          7s  x 2688 -> 5h  | 0.5h  |
# version       7s  x 3024 -> 5h  | 0.5h
# architecture  7s  x 2016 -> 3h  | 0.3h
# template      7s  x 4032 -> 7h  | 0.35h
