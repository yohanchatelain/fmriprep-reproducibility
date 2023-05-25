import argparse
import json


def index(_dict):
    """
    Return a dictionnary indexing the dataset/subject dict
    @_dict: Dictionnary with dataset as key and
            dictionnary with subject as key as value
            ex: {'dataset1' : {'subject1':..., 'subject2':...}, ...}
    @return: {1:('dataset1','subject1'), 2:('dataset1','subject2') }
    """
    _map = dict()
    i = 1
    for dataset, subjects in sorted(_dict.items()):
        for subject in sorted(subjects):
            _map[i] = {"dataset": dataset, "subject": subject}
            i += 1

    return _map


def get_index(_dict, dataset, subject):
    """
    returns the index of a given dataset/subejct pair
    """
    _index = index(_dict)
    for idx, _dict in _index.items():
        _dataset = _dict["dataset"]
        _subject = _dict["subject"]
        if (_dataset, _subject) == (dataset, subject):
            return idx

    raise IndexError(f"No ({dataset},{subject}) pair found")


def parse_json(args):
    filename = args.input
    with open(filename, "r", encoding="utf-8") as fi:
        dataset_dict = json.load(fi)
        return index(dataset_dict)


def dump(args, _map):
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump(_map, fo)
    else:
        print(json.dumps(_map, indent=4))


def parse_args():
    parser = argparse.ArgumentParser("indexing")
    parser.add_argument("--input", required=True, help="JSON file input")
    parser.add_argument("--output", help="output filename")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    _map = parse_json(args)
    dump(args, _map)


if "__main__" == __name__:
    main()
