import pickle
import sys
import glob


def get_path(input_prefix):
    return glob.glob(f'{input_prefix}/*.pkl')


def get_fwh(filename):
    return int(filename.split('fwh_')[-1].split('.pkl')[0])


def add_fwh(filename):
    fwh = get_fwh(filename)
    with open(filename, 'rb') as fi:
        pkl = pickle.load(fi)
        for elt in pkl:
            elt['fwh'] = fwh

    return pkl


def dump(input_prefix, output_prefix, filename, pkl):
    output = filename.replace(input_prefix, output_prefix)
    print(f'{filename} -> {output}')
    with open(output, 'wb') as fo:
        pickle.dump(pkl, fo)


def main():
    input_prefix = sys.argv[1]
    output_prefix = sys.argv[2]

    paths = get_path(input_prefix)
    for path in paths:
        pkl = add_fwh(path)
        dump(input_prefix, output_prefix, path, pkl)


if __name__ == '__main__':
    main()
