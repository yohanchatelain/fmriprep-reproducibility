import scipy
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Binomial distribution
def binomial_sf(k, n, p):
    return 1 - scipy.stats.binom.cdf(k, n, p)


n = 30
k = np.arange(0, n + 1)

alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

fig = go.Figure()

for p in alpha:
    y = binomial_sf(k, n, 1 - p)
    fig.add_trace(
        go.Scatter(x=k, y=y, name=f"1-alpha = {1-p:.3f}", mode="lines+markers")
    )

fig.add_hline(y=0.05, line_dash="dash", line_color="black")
fig.update_xaxes(title="Number of successes k")
fig.update_yaxes(title="Probability to have at least k successes")
fig.show()
