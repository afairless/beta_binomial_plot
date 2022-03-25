#! usr/bin/env python3

import numpy as np
from scipy.stats import beta
from scipy.stats import binom

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output


def generate_binomial_data() -> tuple[float, np.ndarray]:
    """
    Generates a sample of binomial data
    """

    p = 0.5
    binomial_sample = binom.rvs(1, p, size=100, random_state=13833)

    return p, binomial_sample  


def read_beta_priors() -> tuple:

    return 5, 2


def calculate_beta_param_series() -> list[tuple[int | float]]:

    _, binomial_sample = generate_binomial_data()
    beta_params = read_beta_priors()
    alpha_param_sums = binomial_sample.cumsum() 
    beta_param_sums = range(len(alpha_param_sums)) - alpha_param_sums + 1
    beta_param_series = [
        (beta_params[0] + alpha_param_sums[i], 
         beta_params[1] + beta_param_sums[i]) 
        for i in range(len(binomial_sample))]
    beta_param_series = [beta_params] + beta_param_series

    return beta_param_series


def beta_statistical_attributes() -> tuple[np.ndarray, float, int]:

    # the beta distribution is defined over the interval [0, 1]
    x = np.arange(0, 1.01, 0.01)

    threshold = 0.50
    idx50 = int(threshold * len(x))

    return x, threshold, idx50 


app = dash.Dash()

# layout from:  https://community.plotly.com/t/two-graphs-side-by-side/5312/2
app.layout = dash.html.Div([

    dash.html.Div([dash.html.H1(id='heading', style={'textAlign': 'center'})]),

    # add some extra space between title and elements below it
    #dash.html.Div([dash.html.H1(id='placeholder', style={'color': 'white'})]),

    dash.html.Div([
        dash.html.Div(
            children=[dash.dcc.Graph(
                id='beta_plot_01', 
                style={'width': '80vh', 'height': '80vh'})], 
            className="four columns"),
        dash.html.Div(
            children=[dash.dcc.Graph(
                id='beta_plot_02', 
                style={'width': '80vh', 'height': '80vh'})], 
            className="offset-by-two four columns"),
    ], className="row"),

    dash.dcc.Interval(id='interval-component', interval=500, n_intervals=0)
])

# this CSS file placed inside 'assets' directory within the dash app directory
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# reportedly an updated way to load an external CSS; not clear whether it works
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


@app.callback(
    Output('heading', 'children'),
    Input('interval-component', 'n_intervals'))
def overall_heading(n_intervals: int):
    beta_param_series = calculate_beta_param_series()
    loop_len = min(n_intervals+1, len(beta_param_series))
    beta_params = beta_param_series[loop_len-1]
    title = f'Alpha = {beta_params[0]}, Beta = {beta_params[1]}'
    return title


@app.callback(
    Output('beta_plot_01', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_beta_01(n_intervals: int):

    true_beta_mode, binomial_sample = generate_binomial_data()
    beta_param_series = calculate_beta_param_series()
    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    max_line_width = 5
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    fig.add_vline(x=true_beta_mode)

    loop_len = min(n_intervals+1, len(beta_param_series))
    for i in range(loop_len):

        # make the most recently added lines/traces thicker than older traces
        line_width = [
            max(1, max_line_width - (loop_len - j - 1)) 
            for j in range(loop_len)]
        line01 = {'color': trace_color, 'width': line_width[i]}

        # make the most recently added lines/traces more opaque than older traces
        line_opacity = np.linspace(0.3, 1, loop_len)
        if len(line_opacity) == 1:
            line_opacity = [1]

        # calculate density for each beta distribution
        beta_param_alpha = beta_param_series[i][0]
        beta_param_beta = beta_param_series[i][1]
        y = beta.pdf(x, beta_param_alpha, beta_param_beta)

        fig.add_trace(go.Scatter(x=x, y=y, line=line01, opacity=line_opacity[i]))

    # vertical line at 0 or 1 to indicate most recently added data point
    if n_intervals > 0 and binomial_sample[loop_len-2] == 0:
        fig.add_vline(x=0, line_color='green', line_width=4)
    elif n_intervals > 0 and binomial_sample[loop_len-2] == 1:
        fig.add_vline(x=1, line_color='blue', line_width=4)

    beta_mode = (beta_param_alpha - 1) / (beta_param_alpha + beta_param_beta - 2)
    fig.add_vline(x=beta_mode, line_color=trace_color)
    beta_mode = round(beta_mode, 2)
    title = f'True mode = {true_beta_mode}<br><span style="color:red">Mode estimate= {beta_mode}</span>'

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black')
    fig.update_layout(title=title, title_x=0.5)

    return fig


@app.callback(
    Output('beta_plot_02', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_beta_02(n_intervals: int):

    true_beta_mode, binomial_sample = generate_binomial_data()
    beta_param_series = calculate_beta_param_series()
    loop_len = min(n_intervals+1, len(beta_param_series))
    beta_params = beta_param_series[loop_len-1]

    x, threshold, idx50 = beta_statistical_attributes()

    left_color = 'green'
    right_color = 'blue'
    line01 = {'color': left_color, 'width': 5}
    line02 = {'color': right_color, 'width': 5}

    title = ('Proportional area under curve<br>'
        f'<span style="color:{left_color}">beta_prop_text0</span>     '
        f'<span style="color:{right_color}">beta_prop_text1</span>')

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))

    title = title.replace('beta_prop_text0', beta_prop_text0)
    title = title.replace('beta_prop_text1', beta_prop_text1)

    fig = go.Figure(layout=layout)
    fig.add_vline(x=true_beta_mode)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))

    # vertical line at 0 or 1 to indicate most recently added data point
    if n_intervals > 0 and binomial_sample[loop_len-2] == 0:
        fig.add_vline(x=0, line_color=left_color, line_width=4)
    elif n_intervals > 0 and binomial_sample[loop_len-2] == 1:
        fig.add_vline(x=1, line_color=right_color, line_width=4)

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=True)
    fig.update_layout(title=title, title_x=0.5)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
