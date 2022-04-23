#! usr/bin/env python3

import numpy as np
from scipy.stats import beta
from scipy.stats import binom

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash_daq import BooleanSwitch


def generate_binomial_data(
    true_beta_mode: float, data_points_n: int, 
    random_state: int) -> np.ndarray:
    """
    Generates a sample of binomial data
    """

    # if user provides no input values, assign default value
    if not data_points_n:
        data_points_n = 100
    if not random_state:
        random_state = 2424313

    binomial_sample = binom.rvs(
        1, true_beta_mode, size=data_points_n, random_state=random_state)

    return binomial_sample  


def beta_statistical_attributes() -> tuple[np.ndarray, float, int]:
    """
    Defines and returns attributes used for the beta distribution plots
    """

    # the beta distribution is defined over the interval [0, 1]
    x = np.arange(0, 1.01, 0.01)

    # binary decision threshold
    threshold = 0.50

    # index of decision threshold on the 'x' axis range 
    idx50 = int(threshold * len(x))

    return x, threshold, idx50 


def calculate_beta_parameter_series(
    beta_parameter_alpha: float, 
    beta_parameter_beta: float, 
    binomial_sample: np.ndarray) -> list[tuple[float, float]]:
    """
    Given the prior beta distribution parameters alpha and beta and a binomially
        distributed sample, calculate the series of beta parameters for each
        update of the beta distribution based on each successive item in the 
        binomial sample
    """

    # if user provides no input values, assign default value
    if not beta_parameter_alpha:
        beta_parameter_alpha = 2
    if not beta_parameter_beta:
        beta_parameter_beta = 2

    alpha_param_sums = binomial_sample.cumsum() 
    beta_param_sums = range(len(alpha_param_sums)) - alpha_param_sums + 1
    beta_param_series = [
        (beta_parameter_alpha + alpha_param_sums[i], 
         beta_parameter_beta + beta_param_sums[i]) 
        for i in range(len(binomial_sample))]
    beta_param_series = (
        [(beta_parameter_alpha, beta_parameter_beta)] + beta_param_series)

    return beta_param_series


# vertical white space between rows of input widgets
padding_space = "15px"

app = dash.Dash()

# layout from:  https://community.plotly.com/t/two-graphs-side-by-side/5312/2
app.layout = dash.html.Div([

    dash.html.Div([
        dash.html.Div(
            children=dash.html.Button(
                children='Restart Animation', 
                id='interval_reset', 
                n_clicks=0), 
            className="two columns"),
        dash.html.Div(
            children=BooleanSwitch(
                id='pause-toggle', 
                on=True, 
                label='Play Animation'), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.P('Animation Update Frequency (sec)'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='update-frequency', min=0.25, max=10, step=0.25, value=1, 
                marks={i: str(i) for i in range(1, 11)},
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'),
            className="six columns"),
    ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Beta parameter alpha starting value (prior)')), 
            className="four columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Beta parameter beta starting value (prior)')), 
            className="four columns"),
        ], className="row"),

    dash.html.Div([
        dash.html.Div(
            children=dash.dcc.Input(
                id='beta_parameter_alpha', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha'), 
            className="four columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='beta_parameter_beta', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Beta'), 
            className="four columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Number of binomially distributed data points')), 
            className="four columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Random seed')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Binomial Proportion (True Beta Distribution Mode)')), 
            className="four columns"),
        ], className="row"),

    dash.html.Div([
        dash.html.Div(
            children=dash.dcc.Input(
                id='data_points_n', type='number', 
                min=3, max=1000, step=1, value=100, 
                placeholder='Number of Data Points'), 
            className="four columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='random_state', type='number', 
                min=101, max=1e8, step=1, value=1e4, placeholder='Random Seed'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='binomial_proportion', 
                min=0, max=1, step=0.01, value=0.5, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="four columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([dash.html.H1(id='heading', style={'textAlign': 'center'})]),

    dash.html.Div([
        dash.html.Div(
            children=[dash.dcc.Graph(
                id='beta_plot_01', 
                style={'width': '80vh', 'height': '60vh'})], 
            className="four columns"),
        dash.html.Div(
            children=[dash.dcc.Graph(
                id='beta_plot_02', 
                style={'width': '80vh', 'height': '60vh'})], 
            className="offset-by-two four columns"),
    ], className="row"),

    dash.dcc.Interval(id='interval-component', interval=1_000, n_intervals=0, disabled=False)
])

# this CSS file placed inside 'assets' directory within the dash app directory
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


@app.callback(
    Output('interval-component', 'disabled'),
    Input('pause-toggle', 'on'))
def pause_animation(enabled: bool) -> bool:
    """
    Boolean user choice of whether to play or pause the animation of updating 
        beta distributions
    """
    return not enabled


@app.callback(
    Output('interval-component', 'interval'),
    Input('update-frequency', 'value'))
def animation_frequency(value: float) -> float:
    """
    User choice of interval specifying how often the animation should be updated
    User chooses value in seconds, which is converted to milliseconds
    """
    return value * 1000


@app.callback(
    Output('interval-component', 'n_intervals'),
    [Input('interval_reset', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    State('interval_reset', 'value'))
def reset_interval(n_clicks, n_intervals, value) -> int:
    """
    User choice to restart animation from the start of the beta update sequence
    """
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'interval_reset' in changed_id:
        return 0
    else:
        return n_intervals


@app.callback(
    Output('heading', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('beta_parameter_alpha', 'value'), 
     Input('beta_parameter_beta', 'value'),
     Input('binomial_proportion', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def overall_heading(
    n_intervals: int, beta_parameter_alpha: float, 
    beta_parameter_beta: float, true_beta_mode: float, data_points_n: int, 
    random_state: int) -> go.Figure:
    """
    Displays beta distribution parameters alpha and beta of most recent update
    """

    binomial_sample = generate_binomial_data(
        true_beta_mode, data_points_n, random_state)
    beta_parameter_series = calculate_beta_parameter_series(
        beta_parameter_alpha, beta_parameter_beta, binomial_sample)
    loop_len = min(n_intervals+1, len(beta_parameter_series))
    beta_parameters = beta_parameter_series[loop_len-1]
    title = f'Alpha = {beta_parameters[0]}, Beta = {beta_parameters[1]}'
    return title


@app.callback(
    Output('beta_plot_01', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('beta_parameter_alpha', 'value'), 
     Input('beta_parameter_beta', 'value'),
     Input('binomial_proportion', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_beta_01(
    n_intervals: int, beta_parameter_alpha: float, 
    beta_parameter_beta: float, true_beta_mode: float, data_points_n: int, 
    random_state: int) -> go.Figure:
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    Each binomial sample update (a 'zero' or a 'one') is displayed as a vertical
        line at the appropriate value on the x-axis
    """

    binomial_sample = generate_binomial_data(
        true_beta_mode, data_points_n, random_state)
    beta_param_series = calculate_beta_parameter_series(
        beta_parameter_alpha, beta_parameter_beta, binomial_sample)
    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 20})
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
        beta_parameter_alpha = beta_param_series[i][0]
        beta_parameter_beta = beta_param_series[i][1]
        y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)

        fig.add_trace(go.Scatter(x=x, y=y, line=line01, opacity=line_opacity[i]))

    # vertical line at 0 or 1 to indicate most recently added data point
    if n_intervals > 0 and binomial_sample[loop_len-2] == 0:
        fig.add_vline(x=0, line_color='green', line_width=4)
    elif n_intervals > 0 and binomial_sample[loop_len-2] == 1:
        fig.add_vline(x=1, line_color='blue', line_width=4)

    beta_mode = (
        (beta_parameter_alpha - 1) / 
        (beta_parameter_alpha + beta_parameter_beta - 2))

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
    [Input('interval-component', 'n_intervals'),
     Input('beta_parameter_alpha', 'value'), 
     Input('beta_parameter_beta', 'value'),
     Input('binomial_proportion', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_beta_02(
    n_intervals: int, beta_parameter_alpha: float, 
    beta_parameter_beta: float, true_beta_mode: float, data_points_n: int, 
    random_state: int) -> go.Figure:
    """
    Plots most recent beta distribution update with a division marked by the
        decision threshold
    Specifies the proportions of the areas under the beta distribution curve on 
        each side of the decision threshold
    Each binomial sample update (a 'zero' or a 'one') is displayed as a vertical
        line at the appropriate value on the x-axis
    """

    binomial_sample = generate_binomial_data(
        true_beta_mode, data_points_n, random_state)
    beta_parameter_series = calculate_beta_parameter_series(
        beta_parameter_alpha, beta_parameter_beta, binomial_sample)
    loop_len = min(n_intervals+1, len(beta_parameter_series))
    beta_parameters = beta_parameter_series[loop_len-1]

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
        'font_size': 20})

    y = beta.pdf(x, beta_parameters[0], beta_parameters[1])

    beta_prop0 = beta.cdf(threshold, beta_parameters[0], beta_parameters[1])
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
    app.run_server(debug=False, use_reloader=True)
