
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd
import sys
import glob
import os
import re

# non-normal voxel ratio   = 1.89e-01 [18.907003 %]


def parse_line(line):
    r = re.compile(r"\[(.*)%\]")
    [pct] = r.findall(line)
    return float(pct)


def parse_file(filename):
    with open(filename, 'r') as fi:
        for line in fi:
            if line.startswith('non-normal voxel ratio'):
                return parse_line(line)
    print(filename)


# non-normal-rr-ds001748_sub-adult15_7_0.100.log
def get_params(filename):
    head, subject, fwh, tail = filename.split('_')
    dataset = head.split('-')[-1]
    alpha = tail.split('.log')[0]
    return (dataset, subject, fwh, alpha)


def get_log(directory):
    # {dataset-subejct : {fwh : [(ratio1,alpha1), ..., (ratio_n, alpha_n)] } }
    files = glob.glob(os.path.join(directory, 'non-normal-*.log'))
    df = pd.DataFrame(columns=['dataset', 'subject', 'fwh', 'alpha', 'ratio'])
    for file in files:
        (dataset, subject, fwh, alpha) = get_params(file)
        ratio = parse_file(file)
        if ratio is None:
            continue
        df.loc[-1] = [dataset, subject, float(fwh), float(alpha), float(ratio)]
        df.index += 1

    df.sort_index(inplace=True)
    return df


def plot_facet(df):
    print('Plot facet')
    fig = px.scatter(df, x='alpha', y='ratio', color='fwh',
                     color_continuous_scale='Jet', facet_col='subject')

    fig.update_layout(
        title='% of voxels rejecting Shapiro-Wilk normality test')
    fig.update_yaxes(range=[-1, 101])
    fig.write_image('facet.png', scale=2)
    fig.write_html('facet.html')


def plot_subject(df):
    subject = df['subject'].unique()[0]
    print(f'Plot {subject}')
    fig = px.scatter(df, x='alpha', y='ratio',
                     color='fwh', color_continuous_scale='Jet')
    nb_fwh = df['fwh'].unique().size
    nb_alpha = df['alpha'].unique().size

    alpha = df['alpha'].unique()
    alpha.sort()
    alpha_line = go.Scatter(x=alpha, y=100 * alpha, mode='lines', line=dict(color='black'),
                            name='nominal value')

    shift = np.linspace(0.99, 1.11, nb_fwh)
    fig.data[0]['x'] *= np.repeat(shift, nb_alpha)
    fig.update_yaxes(range=[-1, 101])

    fig.add_trace(alpha_line)

    fig.update_layout(
        title=f'% of voxels rejecting Shapiro-Wilk normality test ({subject})')
    fig.write_image(f'{subject}.png', scale=2)
    fig.write_html(f'{subject}.html')


def plot(df):
    df.sort_values(by=['dataset', 'subject', 'fwh', 'alpha'], inplace=True)
    plot_facet(df)
    subjects = df['subject'].unique()
    for subject in subjects:
        try:
            plot_subject(df[df['subject'] == subject])
        except:
            continue


if __name__ == '__main__':
    directory = sys.argv[1]
    d = get_log(directory)
    d.to_csv('non-normal-log.csv')
    plot(d)
