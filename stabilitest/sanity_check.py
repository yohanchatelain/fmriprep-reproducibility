import os
import argparse
import subprocess

SUCCESS_STATUS = "success"
FAIL_STATUS = "fail"

MRI_PICKLE_DIR = 'mri_pickle'
MRI_LOG = 'mri_log'

confidences = [0.75, 0.8, 0.9, 0.95, 0.99, 0.995]

datasets = {"ds000256": ["sub-CTS201", "sub-CTS210"],
            "ds001748": ["sub-adult15", "sub-adult16"],
            "ds002338": ["sub-xp207", "sub-xp201"],
            "ds001600": ["sub-1"]
            }  # [ds001771]=(sub-36) )


def run_mri_stats(test, confidence, template,
                  reference_prefix, reference_dataset, reference_subject,
                  target_prefix, target_dataset, target_subject,
                  output, status):

    output = f'{MRI_PICKLE_DIR}/{output}'
    log_filename = f'{MRI_LOG}/{output}.log'
    log_fo = open(log_filename, 'w')
    err_filename = f'{MRI_LOG}/{output}.err'
    err_fo = open(err_filename, 'w')

    cmd = f'python3 MRI-stats/__main__.py {test} '
    args = [f'--confidence={confidence}',
            '--template=MNI152NLin2009cAsym',
            '--data-type=anat',
            f'--reference-prefix={reference_prefix}',
            f'--reference-dataset={reference_dataset}',
            f'--reference-subject={reference_subject} ',
            f'--output={output}']

    if target_prefix:
        args.append(f'--reference-prefix={reference_prefix}')
    if target_dataset:
        args.append(f'--reference-dataset={reference_dataset}')
    if target_subject:
        args.append(f'--reference-subject={reference_subject}')

    bin_ = cmd + ' ' + ' '.join(args)

    subprocess.check_call(bin_, stdout=log_fo, stderr=err_fo, shell=True)

    pickle_filename = "{MRI_PICKLE_DIR}/{output}.pkl"
    cmd = "python3 mri_check_status.py"
    args = ['--status={status}', f'--filename={pickle_filename}']
    bin_ = cmd + ' ' + ' '.join(args)
    subprocess.check_call(bin_, shell=True)


def run_test_all_include(reference_prefix,
                         reference_dataset,
                         reference_subject,
                         reference_name,
                         target_prefix,
                         target_dataset,
                         target_subject,
                         target_name,
                         status,
                         confidence):

    test = 'all-include'
    output = '_'.join([test, confidence, 'reference', reference_name,
                       reference_dataset, reference_subject])
    run_mri_stats(test=test,
                  confidence=confidence,
                  reference_prefix=reference_prefix,
                  reference_dataset=reference_dataset,
                  reference_subject=reference_subject,
                  output=output,
                  status=status)


def run_test_all_exclude(reference_prefix,
                         reference_dataset,
                         reference_subject,
                         reference_name,
                         target_prefix,
                         target_dataset,
                         target_subject,
                         target_name,
                         status,
                         confidence):

    test = 'all-exclude'
    output = '_'.join([test, confidence, 'reference', reference_name,
                       reference_dataset, reference_subject])
    run_mri_stats(test=test,
                  confidence=confidence,
                  reference_prefix=reference_prefix,
                  reference_dataset=reference_dataset,
                  reference_subject=reference_subject,
                  output=output,
                  status=status)


def run_test_one(reference_prefix,
                 reference_dataset,
                 reference_subject,
                 reference_name,
                 target_prefix,
                 target_dataset,
                 target_subject,
                 target_name,
                 status,
                 confidence):

    test = 'one'
    output = '_'.join([test, confidence, 'reference', reference_name,
                       reference_dataset, reference_subject,
                       'target', target_name, target_dataset, target_subject])

    run_mri_stats(test=test, confidence=confidence,
                  reference_prefix=reference_prefix,
                  reference_dataset=reference_dataset,
                  reference_subject=reference_subject,
                  target_prefix=target_prefix,
                  target_dataset=target_dataset,
                  target_subject=target_subject,
                  output=output,
                  status=status)


def run_test_inter(reference_prefix,
                   reference_dataset,
                   reference_subject,
                   reference_name,
                   target_prefix,
                   target_dataset,
                   target_subject,
                   target_name,
                   status,
                   confidence):

    status = FAIL_STATUS
    for dataset, subjects in datasets.items():
        for subject in subjects:
            if subject == reference_subject:
                continue

            run_test_one(reference_prefix=reference_prefix,
                         reference_dataset=reference_dataset,
                         reference_subject=reference_subject,
                         reference_name=reference_name,
                         target_prefix=reference_prefix,
                         target_dataset=dataset,
                         target_subject=subject,
                         target_name=reference_name,
                         status=status,
                         confidence=confidence)


tests = {
    'all-include': run_test_all_include,
    'all-exclude': run_test_all_exclude,
    'one': run_test_one,
    'inter': run_test_inter
}


def run(margs, *args, **kwargs):
    for confidence in confidences:
        for dataset, subjects in datasets.items():
            for subject in subjects:
                tests[margs.test](*args, **kwargs,
                                  status=SUCCESS_STATUS,
                                  reference_dataset=dataset,
                                  reference_subject=subject,
                                  target_dataset=dataset,
                                  target_subject=subject,
                                  confidence=confidence)


def parse_args():
    parser = argparse.ArgumentParser('sanity-check')
    parser.add_argument('--reference-prefix')
    parser.add_argument('--reference-name')
    parser.add_argument('--target-prefix')
    parser.add_argument('--target-name')
    parser.add_argument('--test', nargs='+', required=True,
                        choices=list(tests.keys()))
    return parser.parse_args()


def set_dirs():
    os.makedirs(MRI_LOG, exist_ok=True)
    os.makedirs(MRI_PICKLE_DIR, exist_ok=True)


def main():
    args = parse_args()
    run(args, reference_prefix=args.reference_prefix,
        reference_name=args.reference_name,
        target_prefix=args.target_prefix,
        target_name=args.target_name)


if __name__ == '__main__':
    set_dirs()
    main()
