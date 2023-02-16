import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv('stats.csv')
df = df.drop('Unnamed: 0', axis=1)
rr = df[df['prefix'] == 'rr']
print(rr)
rs = df[df['prefix'] == 'rs']
rr_rs = df[df['prefix'] == 'rr.rs']

fig = make_subplots(rows=3, cols=3, shared_xaxes=True)

# mean
fig = px.scatter(rr[rr.stat == 'mean'], x='fwh', y='mean',
                 color='subject', title='RR mean (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-10, 6000])
fig.write_image('rr_mean_mean.pdf')

fig = px.scatter(rs[rs.stat == 'mean'], x='fwh', y='mean',
                 color='subject', title='RS mean (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-10, 6000])
fig.write_image('rs_mean_mean.pdf')

fig = px.scatter(rr_rs[rr_rs.stat == 'mean'], x='fwh', y='mean',
                 color='subject', title='RR+RS mean (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-10, 6000])
fig.write_image('rr-rs_mean_mean.pdf')

# std
fig = px.scatter(rr[rr.stat == 'std'], x='fwh', y='mean',
                 color='subject', title='RR std (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 60])
fig.write_image('rr_std_mean.pdf')

fig = px.scatter(rs[rs.stat == 'std'], x='fwh', y='mean',
                 color='subject', title='RS std (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 60])
fig.write_image('rs_std_mean.pdf')

fig = px.scatter(rr_rs[rr_rs.stat == 'std'], x='fwh', y='mean',
                 color='subject', title='RR+RS std (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 60])
fig.write_image('rr-rs_std_mean.pdf')

# sig
fig = px.scatter(rr[rr.stat == 'sig'], x='fwh', y='mean',
                 color='subject', title='RR sig (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 11])
fig.write_image('rr_sig_mean.pdf')

fig = px.scatter(rs[rs.stat == 'sig'], x='fwh', y='mean',
                 color='subject', title='RS sig (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 11])
fig.write_image('rs_sig_mean.pdf')


fig = px.scatter(rr_rs[rr_rs.stat == 'sig'], x='fwh', y='mean',
                 color='subject', title='RR+RS sig (mean)')
fig.update_traces(marker=dict(size=10), mode='markers')
fig.update_yaxes(range=[-3, 11])
fig.write_image('rr-rs_sig_mean.pdf')
