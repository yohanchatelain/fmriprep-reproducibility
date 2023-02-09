
import sys
import glob
import os
import re

# non-normal voxel ratio   = 1.89e-01 [18.907003 %]


def parse_line(line):
    r = re.compile(r"\[(.*) %\]")
    [pct] = r.findall(line)
    return float(pct)


def parse_file(filename):
    with open(filename, 'r') as fi:
        for line in fi:
            if line.startswith('non-normal voxel ratio'):
                return parse_line(line)
    raise Exception('No #voxels line found')


# non-normal-rr-ds001748_sub-adult15_7_0.100.log
def get_params(filename):
    head, subject, fwh, tail = filename.split('_')
    dataset = head.split('-')[-1]
    alpha = tail.split('.')[0]
    return (dataset, subject, fwh, alpha)


def get_log(directory):
    # {dataset-subejct : {fwh : [(ratio1,alpha1), ..., (ratio_n, alpha_n)] } }
    files = glob.glob(os.path.join(directory, 'non-normal-*.log'))
    d = {}
    for file in files:
        (dataset, subject, fwh, alpha) = get_params(file)
        ratio = parse_file(file)
        key = f'{dataset} {subject}'
        d[key] = d.get(key, []) + [(alpha, ratio)]

    return d


if __name__ == '__main__':
    directory = sys.argv[1]
    d = get_log(directory)
    print(d)
