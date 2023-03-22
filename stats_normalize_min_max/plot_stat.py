import plotly
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.kaleido.scope.mathjax = None


def show_named_plotly_colours():
    """
    function to display to user the colours to match plotly's named
    css colours.

    Reference:
        #https://community.plotly.com/t/plotly-colours-list/11730/3

    Returns:
        plotly dataframe with cell colour to match named colour name

    """
    s = '''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
    li = s.split(',')
    li = [l.replace('\n', '') for l in li]
    li = [l.replace(' ', '') for l in li]

    import pandas as pd
    import plotly.graph_objects as go

    df = pd.DataFrame.from_dict({'colour': li})
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Plotly Named CSS colours"],
            line_color='black', fill_color='white',
            align='center', font=dict(color='black', size=14)
        ),
        cells=dict(
            values=[df.colour],
            line_color=[df.colour], fill_color=[df.colour],
            align='center', font=dict(color='black', size=11)
        ))
    ])

    fig.show()


show_named_plotly_colours()

df = pd.read_csv('stats.csv')
df = df.drop('Unnamed: 0', axis=1)
rr = df[df['prefix'] == 'rr']
rs = df[df['prefix'] == 'rs']
rr_rs = df[df['prefix'] == 'rr.rs']
pd.set_option('display.max_rows', 600)

rr.sort_values(by=['fwh', 'stat', 'subject'], axis=0, inplace=True)
rs.sort_values(by=['fwh', 'stat', 'subject'], axis=0, inplace=True)
rr_rs.sort_values(by=['fwh', 'stat', 'subject'], axis=0, inplace=True)

fig = make_subplots(rows=3, cols=3,
                    shared_xaxes=True, shared_yaxes=True,
                    vertical_spacing=0.01, horizontal_spacing=0.01,
                    column_titles=['RR', 'RS', 'RR+RS'],
                    row_titles=['Mean', 'Standard deviation',
                                'Significant bits'],
                    x_title='FWHM (mm)')

color = plotly.colors.make_colorscale(['darkblue', 'blue', 'cyan',
                                       'green', 'yellow', 'orange',
                                       'red', 'darkred'])
subjects = {s: s for s in rr['subject'].unique()}
colors = {s: color[i] for i, s in enumerate(subjects.keys())}

# mean
sc = px.scatter(rr[rr.stat == 'mean'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_layout(showlegend=False)
sc.update_yaxes(range=[-10, 6000])

fig.add_traces(sc.data, rows=1, cols=1)

sc = px.scatter(rs[rs.stat == 'mean'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_yaxes(range=[-10, 6000])
fig.add_traces(sc.data, rows=1, cols=2)

sc = px.scatter(rr_rs[rr_rs.stat == 'mean'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_yaxes(range=[-10, 6000])
fig.add_traces(sc.data, rows=1, cols=3)

# std
sc = px.scatter(rr[rr.stat == 'std'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_yaxes(range=[-3, 60])
fig.add_traces(sc.data, rows=2, cols=1)

sc = px.scatter(rs[rs.stat == 'std'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_yaxes(range=[-3, 60])
fig.add_traces(sc.data, rows=2, cols=2)

sc = px.scatter(rr_rs[rr_rs.stat == 'std'], x='fwh', y='mean',
                color='subject',
                category_orders=subjects,
                log_y=False)
sc.update_yaxes(range=[-3, 60])
fig.add_traces(sc.data, rows=2, cols=3)

# sig
sc = px.scatter(rr[rr.stat == 'sig'], x='fwh', y='mean',
                color='subject', title='RR sig (mean)',
                category_orders=subjects)
sc.update_yaxes(range=[-3, 11])
fig.add_traces(sc.data, rows=3, cols=1)

sc = px.scatter(rs[rs.stat == 'sig'], x='fwh', y='mean',
                color='subject', title='RS sig (mean)',
                category_orders=subjects)
sc.update_yaxes(range=[-3, 11])
fig.add_traces(sc.data, rows=3, cols=2)

sc = px.scatter(rr_rs[rr_rs.stat == 'sig'], x='fwh', y='mean',
                color='subject', title='RR+RS sig (mean)',
                category_orders=subjects)
sc.update_yaxes(range=[-3, 11])
fig.add_traces(sc.data, rows=3, cols=3)

fig.update_traces(marker=dict(size=3), mode='lines+markers')

# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=1, col=1)
# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=1, col=2)
# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=1, col=3)

# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=2, col=1)
# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=2, col=2)
# fig.update_yaxes(type='log', showexponent='all', dtick="D2",
#                  exponentformat='power', row=2, col=3)

print(fig.data[0]['name'])  # sub-1
print(fig.data[1]['name'])  # 36
print(fig.data[2]['name'])  # CTS201
print(fig.data[3]['name'])  # CTS210
print(fig.data[4]['name'])  # adult15
print(fig.data[5]['name'])  # adult16
print(fig.data[6]['name'])  # xp201
print(fig.data[7]['name'])  # xp207

for i, d in enumerate(fig.data):
    if i % 8 == 0:
        print('-' * 20)
    print(i, d['name'])

for i in range(9):
    fig.data[5 + i * 8]['marker']['color'] = 'darkblue'
    fig.data[4 + i * 8]['marker']['color'] = 'dodgerblue'
    fig.data[0 + i * 8]['marker']['color'] = 'mediumturquoise'
    fig.data[6 + i * 8]['marker']['color'] = 'limegreen'
    fig.data[7 + i * 8]['marker']['color'] = 'orange'
    fig.data[1 + i * 8]['marker']['color'] = 'darkorange'
    fig.data[2 + i * 8]['marker']['color'] = 'tomato'
    fig.data[3 + i * 8]['marker']['color'] = 'firebrick'

print(len(fig.data))

fig.update_coloraxes(showscale=False,  row=1, col=1)
sc.update_layout(showlegend=False)

for d in fig.data[8:]:
    d['showlegend'] = False

fig.update_layout(font=dict(size=8),
                  legend=dict(orientation='h',
                              yanchor='bottom',
                              xanchor='left',
                              # y=0.01,
                              bgcolor='rgba(0,0,0,0)'),
                  margin=dict(l=5, r=7, b=10, t=25))

fig.update_annotations(font=dict(size=10))

print(fig)
fig.write_image('stats.pdf', scale=10)
# fig.show()
