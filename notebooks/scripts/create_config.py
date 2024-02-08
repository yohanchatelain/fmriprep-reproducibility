import argparse
import json
from typing import Any
import os

from stabilitest.mri_loader.sample import configurator

# Create configuration JSON files to run with stabilitest
# Create a configuration file for each of the following:
# 1. Figure 4: LOO evalutation for RR and RS for 20.2.1
# 2. Figure 5: Ratio of successful runs for within-subject IEEE check for 20.2.1
# 3. Figure 6: Corrupted template check for RR and RS for 20.2.1
# 4. Figure 7: Tests accross different version for RR and RS, 20.2.1 as reference

fmriprep_reference_version = "20.2.1"
fmriprep_versions = [
    "20.2.0",
    "20.2.1",
    "20.2.2",
    "20.2.3",
    "20.2.4",
    "20.2.5",
    "21.0.4",
    "22.1.1",
    "23.0.0",
]
smoothing_kernels = list(range(21))
mask_combination = ["union"]

reference_template = "MNI152NLin2009cAsym"
noised_templates = [
    f"MNI152NLin2009cAsymNoised{lvl}"
    for lvl in ["0001", "0005", "0010", "0050"] + [f"0{i}00" for i in range(1, 10)]
]
reference_architecture = "narval-AMD"
architectures = [
    "cedar-broadwell",
    "cedar-cascade",
    "cedar-skylake",
    "graham-broadwell",
    "graham-cascade",
    "narval-AMD",
]
ieee_prefix = "outputs/20.2.1/ieee"
rr_prefix = "outputs/20.2.1/rr"
rs_prefix = "outputs/20.2.1/rs"


def create_default_configuration():
    """
    Return a default configuration for stabilitest populated with fake values
    """
    return configurator()


def create_reference(
    prefix, dataset, subject, template, version, architecture, perturbation
):
    """
    Create a reference configuration from a list of parameters
    """
    reference = {}
    reference["prefix"] = prefix
    reference["dataset"] = dataset
    reference["subject"] = subject
    reference["template"] = template
    reference["version"] = version
    reference["architecture"] = architecture
    reference["perturbation"] = perturbation
    return reference


def create_target(
    prefix, dataset, subject, template, version, architecture, perturbation
):
    """
    Create a target configuration from a list of parameters
    """
    target = {}
    target["prefix"] = prefix
    target["dataset"] = dataset
    target["subject"] = subject
    target["template"] = template
    target["version"] = version
    target["architecture"] = architecture
    target["perturbation"] = perturbation
    return target


def create_hyperparameters(smoothing_kernels, mask_combination):
    """
    Create a hyperparameters configuration from a list of parameters
    """
    hyperparameters = {}
    hyperparameters["mask-combination"] = mask_combination
    hyperparameters["smooth-kernel"] = smoothing_kernels
    return hyperparameters


def create_configuration(reference, target, hyperparameters, output):
    config: dict[str, Any] = create_default_configuration()
    config["distribution"] = ["normal"]
    config["reference"] = reference
    config["target"] = target
    config["hyperparameters"] = hyperparameters
    config["output"] = output
    return config


def write_configuration(config, output):
    with open(output, "w") as f:
        json.dump(config, f, indent=4)


def create_loo_configuration(inputs_dataset, dry_run=False):
    """
    LOO configuration
    =================
    prefix: variable
    dataset: variable
    subjects: variable
    version: 20.2.1
    template: MNI152NLin2009cAsym
    architecture: narval-AMD
    perturbation: variable
    """
    i = 0
    rr = {"prefix": rr_prefix, "perturbation": "rr"}
    rs = {"prefix": rs_prefix, "perturbation": "rs"}
    template = reference_template
    version = fmriprep_reference_version
    architecture = reference_architecture

    for mode in [rr, rs]:
        prefix = mode["prefix"]
        perturbation = mode["perturbation"]
        for dataset, subjects in inputs_dataset.items():
            for subject in subjects:
                output_json = f"loo_{i}.json"
                output_pkl = f"loo_{i}.pkl"
                i += 1
                reference = create_reference(
                    prefix,
                    dataset,
                    subject,
                    template,
                    version,
                    architecture,
                    perturbation,
                )
                target = create_target(
                    prefix,
                    dataset,
                    subject,
                    template,
                    version,
                    architecture,
                    perturbation,
                )
                hyperparameters = create_hyperparameters(
                    smoothing_kernels, mask_combination
                )
                config = create_configuration(
                    reference, target, hyperparameters, output_pkl
                )
                if dry_run:
                    print(json.dumps(config, indent=2))
                else:
                    output_dir = "configs/loo"
                    os.makedirs(output_dir, exist_ok=True)
                    write_configuration(config, os.path.join(output_dir, output_json))


def create_ieee_subjects_comparison_configuration(inputs_dataset, dry_run=False):
    """
    IEEE subjects comparison configuration
    ======================================
    prefix: variable
    dataset: variable
    subjects: variable
    version: 20.2.1
    template: MNI152NLin2009cAsym
    architecture: narval-AMD
    perturbation: variable
    """
    i = 0
    rr = {"prefix": rr_prefix, "perturbation": "rr"}
    rs = {"prefix": rs_prefix, "perturbation": "rs"}
    template = reference_template
    version = fmriprep_reference_version
    architecture = reference_architecture

    for mode in [rr, rs]:
        prefix = mode["prefix"]
        perturbation = mode["perturbation"]
        for dataset1, subjects1 in inputs_dataset.items():
            for subject1 in subjects1:
                for dataset2, subjects2 in inputs_dataset.items():
                    for subject2 in subjects2:
                        output_json = f"ieee_subjects_comparison_{i}.json"
                        output_pkl = f"ieee_subjects_comparison_{i}.pkl"
                        i += 1
                        reference = create_reference(
                            prefix,
                            dataset1,
                            subject1,
                            template,
                            version,
                            architecture,
                            perturbation,
                        )
                        target = create_target(
                            "outputs/20.2.1/ieee",
                            dataset2,
                            subject2,
                            template,
                            version,
                            architecture,
                            "ieee",
                        )
                        hyperparameters = create_hyperparameters(
                            smoothing_kernels, mask_combination
                        )
                        config = create_configuration(
                            reference, target, hyperparameters, output_pkl
                        )
                        if dry_run:
                            print(json.dumps(config, indent=2))
                        else:
                            output_dir = "configs/ieee/subjects"
                            os.makedirs(output_dir, exist_ok=True)
                            write_configuration(
                                config, os.path.join(output_dir, output_json)
                            )


def create_corrupted_template_configuration(inputs_dataset, dry_run=False):
    """
    Corrupted template configuration
    ======================================
    prefix: variable
    dataset: variable
    subjects: variable
    version: 20.2.1
    template: MNI152NLin2009cAsym
    architecture: narval-AMD
    perturbation: variable
    """
    i = 0
    rr = {"prefix": rr_prefix, "perturbation": "rr"}
    rs = {"prefix": rs_prefix, "perturbation": "rs"}
    version = fmriprep_reference_version
    architecture = reference_architecture

    for mode in [rr, rs]:
        prefix = mode["prefix"]
        perturbation = mode["perturbation"]
        for template in noised_templates:
            for dataset, subjects in inputs_dataset.items():
                for subject in subjects:
                    output_json = f"ieee_template_comparison_{i}.json"
                    output_pkl = f"ieee_template_comparison_{i}.pkl"
                    i += 1
                    reference = create_reference(
                        prefix,
                        dataset,
                        subject,
                        reference_template,
                        version,
                        architecture,
                        perturbation,
                    )
                    target = create_target(
                        "outputs/20.2.1/template/ieee",
                        dataset,
                        subject,
                        template,
                        version,
                        architecture,
                        "ieee",
                    )
                    hyperparameters = create_hyperparameters(
                        smoothing_kernels, mask_combination
                    )
                    config = create_configuration(
                        reference, target, hyperparameters, output_pkl
                    )
                    if dry_run:
                        print(json.dumps(config, indent=2))
                    else:
                        output_dir = "configs/ieee/templates"
                        os.makedirs(output_dir, exist_ok=True)
                        write_configuration(
                            config, os.path.join(output_dir, output_json)
                        )


def create_version_comparison_configuration(inputs_dataset, dry_run=False):
    """
    Version comparison configuration
    ======================================
    prefix: variable
    dataset: variable
    subjects: variable
    version: 20.2.1
    template: MNI152NLin2009cAsym
    architecture: narval-AMD
    perturbation: variable
    """
    i = 0
    rr = {"prefix": rr_prefix, "perturbation": "rr"}
    rs = {"prefix": rs_prefix, "perturbation": "rs"}
    template = reference_template
    architecture = reference_architecture

    for mode in [rr, rs]:
        prefix = mode["prefix"]
        perturbation = mode["perturbation"]
        for version in fmriprep_versions:
            for dataset, subjects in inputs_dataset.items():
                for subject in subjects:
                    output_json = f"ieee_versions_comparison_{i}.json"
                    output_pkl = f"ieee_versions_comparison_{i}.pkl"
                    i += 1
                    reference = create_reference(
                        prefix,
                        dataset,
                        subject,
                        template,
                        fmriprep_reference_version,
                        architecture,
                        perturbation,
                    )
                    target = create_target(
                        f"outputs/{version}/ieee",
                        dataset,
                        subject,
                        template,
                        version,
                        architecture,
                        "ieee",
                    )
                    hyperparameters = create_hyperparameters(
                        smoothing_kernels, mask_combination
                    )
                    config = create_configuration(
                        reference, target, hyperparameters, output_pkl
                    )
                    if dry_run:
                        print(json.dumps(config, indent=2))
                    else:
                        output_dir = "configs/ieee/versions"
                        os.makedirs(output_dir, exist_ok=True)
                        write_configuration(
                            config, os.path.join(output_dir, output_json)
                        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_datasets(inputs):
    with open(inputs, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    dataset = load_datasets(args.inputs)
    create_loo_configuration(dataset, dry_run=args.dry_run)
    create_ieee_subjects_comparison_configuration(dataset, dry_run=args.dry_run)
    create_corrupted_template_configuration(dataset, dry_run=args.dry_run)
    create_version_comparison_configuration(dataset, dry_run=args.dry_run)


if "__main__" == __name__:
    main()
