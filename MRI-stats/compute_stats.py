import glob
import numpy as np
import pandas as pd


def parse_file(filename):
    # rr_ds002338_sub-xp207_MNI152NLin2009cAsym_union_6_std.nii
    fields = filename.split('_')
    print(fields)
    prefix = fields[0]
    dataset = fields[1]
    subject = fields[2]
    template = fields[3]
    mask = fields[4]
    fwh = fields[5]
    stat = fields[6].split('.')[0]
    return {'prefix': prefix, 'dataset': dataset,
            'subject': subject, 'template': template,
            'mask': mask, 'fwh': fwh, 'stat': stat}


def load(df, filename):
    f = parse_file(filename)
    npy = np.load(filename)

    mean = npy.mean()
    std = npy.std()

    df.loc[-1] = [f['prefix'], f['subject'], f['fwh'], f['mask'],
                  f['stat'], mean, std]
    df.index += 1


def get_files():
    return glob.glob('*.npy')


def main():
    files = get_files()
    df = pd.DataFrame(
        columns=['prefix', 'subject', 'fwh', 'mask', 'stat', 'mean', 'std'])
    for f in files:
        load(df, f)
    return df


if '__main__' == __name__:
    df = main()
    df.to_csv('stats.csv')
