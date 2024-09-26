import numpy as np
from numpy.polynomial.polynomial import Polynomial

import torch
from torch import nn

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

DEFAULT_SEED = sum(ord(c) ** 2 for c in "R5.A.12-ModMath")


def torch_rng(rng_seed=DEFAULT_SEED):
    return torch.Generator().manual_seed(rng_seed)


## Linear regression
## ---


def get_dataset(
    params=(5.0, -2.0),
    x_span=(-3.0, 3.0),
    n_data=100,
    noise_amplitude=6.5,
    rng_seed=DEFAULT_SEED,
):
    slope, bias = params
    rng = torch.Generator().manual_seed(rng_seed)
    x = torch.empty(n_data, 1).uniform_(*x_span, generator=rng)
    noise = torch.empty(n_data, 1).normal_(0.0, noise_amplitude, generator=rng)
    y = slope * x + bias + noise
    return x, y


class LinearRegression(nn.Module):
    def __init__(self, w=0.0, b=5.0):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(w))
        self.bias = nn.Parameter(torch.tensor(b))

    def forward(self, x):
        return self.slope * x + self.bias


def build_loss(slope_vals, bias_vals, x, y, vectorized=False):
    loss = np.zeros((len(bias_vals), len(slope_vals)))
    if vectorized:
        y_pred = slope_vals[None, :] * x[..., None, None] + bias_vals[:, None]
        loss = (y_pred - y[..., None, None]) ** 2
        while loss.shape > 2:
            loss = np.mean(loss, axis=0)
    else:
        loss_fun = nn.MSELoss()  # instanciation de la MSE
        for i in range(len(bias_vals)):
            for j in range(len(slope_vals)):
                model_ij = LinearRegression(w=slope_vals[j], b=bias_vals[i])
                y_pred = model_ij(x)
                loss[i, j] = loss_fun(y_pred, y)
    return loss


## Plot utilities
## ---


def linspace_colors(n, colorscale=plotly.colors.sequential.Plasma, low=0.0, high=0.9):
    return plotly.colors.sample_colorscale(colorscale, n, low=low, high=high)


def data_scatter_trace(x, y, mode="markers", **scatter_params):
    return go.Scatter(x=x.flatten(), y=y.flatten(), mode=mode, **scatter_params)


def init_contour_trace(x, y, z, **contour_params):
    default_contour_params = dict(
        contours_coloring="lines",
        colorscale="Greys_r",
        showscale=False,
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        ncontours=40,
        line=dict(smoothing=1.3),
    )
    contour_params = default_contour_params | contour_params
    return go.Contour(x=x, y=y, z=z, **contour_params)


def param_space_evolution(
    w_history,
    b_history,
    colors=None,
    loss_trace=None,
    fig=go.Figure(),
    row=None,
    col=None,
):
    if colors is None:
        colors = linspace_colors(len(w_history))

    if loss_trace is None:
        w_min, w_max = min(w_history), max(w_history)
        b_min, b_max = min(b_history), max(b_history)
        w_range, b_range = w_max - w_min, b_max - b_min
        w = np.linspace(w_min - 0.2 * w_range, w_max + 0.2 * w_range, 30)
        b = np.linspace(b_min - 0.2 * b_range, b_max + 0.2 * b_range, 20)
        loss = build_loss(w, b, *get_dataset())
        loss_trace = init_contour_trace(w, b, loss)

    fig.add_trace(loss_trace, row=row, col=col)
    marker_params = dict(size=10, color=colors)
    scatter_params = dict(marker=marker_params, mode="markers", row=row, col=col)
    fig.add_scatter(x=w_history, y=b_history, **scatter_params)

    fig.update_xaxes(title="w", row=row, col=col)
    fig.update_yaxes(title="b", row=row, col=col)
    return fig


def data_space_evolution(
    w_history,
    b_history,
    colors=None,
    data_trace=None,
    fig=go.Figure(),
    row=None,
    col=None,
):
    if colors is None:
        colors = linspace_colors(len(w_history))

    if data_trace is None:
        x_data, y_data = get_dataset()
        x, y = x_data.flatten().numpy(), y_data.flatten().numpy()
        data_trace = data_scatter_trace(x, y)

    fig.add_trace(data_trace, row=row, col=col)
    x_span = np.array([min(data_trace.x), max(data_trace.x)])
    shared_params = dict(mode="lines", row=row, col=col)
    for w, b, c in zip(w_history, b_history, colors):
        y_span = w * x_span + b
        fig.add_scatter(x=x_span, y=y_span, line=dict(color=c), **shared_params)

    fig.update_xaxes(title="x", row=row, col=col)
    fig.update_yaxes(title="y", row=row, col=col)
    return fig


def training_evolution(w_history, b_history, loss_trace=None, data_trace=None):
    # création de la figure
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.41, 0.59],
        column_titles=[
            "Erreur en fonction des paramètres",
            "Données et régressions linéaires",
        ],
    )
    fig.update_layout(
        width=800, height=350, margin=dict(l=20, r=20, b=20, t=20), showlegend=False
    )

    data_space_evolution(
        w_history, b_history, data_trace=data_trace, fig=fig, row=1, col=2
    )
    param_space_evolution(
        w_history, b_history, loss_trace=loss_trace, fig=fig, row=1, col=1
    )
    return fig
