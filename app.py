import os

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

from interpolation import hermite
from nodes.chebyshev import create_chebyshev_nodes

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H1('Interpolacja funkcji metodą Hermite’a.'),
    html.P('Projekt na WZM'),
    html.Div([
        html.Div([
            html.H6('Funkcja'),
            dcc.Input(
                value='(1+25*x**2)**-1',
                type='text',
                id='function_input'
            ),
            html.H6('Zakres'),
            html.Label('od:'),
            dcc.Input(
                value=-1.0,
                type='number',
                id='begin_input'
            ),
            html.Label('do:'),
            dcc.Input(
                value=1.0,
                type='number',
                id='end_input'
            ),
            html.Label('liczba wezlow(równoodległe, Czebyszewa):'),
            dcc.Input(
                value=10,
                type='number',
                id='num_nodes_input'
            ),
            html.Label('własne węzły:'),
            dcc.Input(
                value='-1, 0, 1',
                type='text',
                id='nodes_input'
            ),
            html.P(''),
            html.Button(
                'Oblicz',
                id='start_button',
            ),
            html.H6('Ustawienia programu:'),
            html.Label('Początek wykresu:'),
            dcc.Input(
                value='-1.1',
                type='text',
                id='start_input'
            ),
            html.Label('Koniec wykresu:'),
            dcc.Input(
                value='1.1',
                type='text',
                id='stop_input'
            ),
            html.Label('Liczba punktów:'),
            dcc.Input(
                value=100,
                type='number',
                id='num_input'
            ),
        ], className="two columns"),
        html.Div([
            html.H6('Wynikowy wielomian dla wezłów równoodległych:'),
            html.Div(
                id='display-value',
                children='xd(x)'
            ),
            html.H6('Wynikowy wielomian dla wezłów Czebyszewa:'),
            html.Div(
                id='display-chebyshev-value',
                children='xd(x)'
            ),
            html.H6('Wynikowy wielomian dla własnych węzłów:'),
            html.Div(
                id='display-own-nodes-value',
                children='xd(x)'
            ),
            html.H6('Wykres funkcji'),
            dcc.Graph(
                id='example-graph',
            )
        ], className="ten columns")
    ], className='row')
])


def get_result_function(nodes, fun_str):
    fun = parse_expr(fun_str)
    x = Symbol('x')
    result = hermite.hermite_interpolationg(nodes, fun, x, True)
    return result

@app.callback(
              dash.dependencies.Output('example-graph', 'figure'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('function_input', 'value'),
               dash.dependencies.State('start_input', 'value'),
               dash.dependencies.State('stop_input', 'value'),
               dash.dependencies.State('num_input', 'value'),
               dash.dependencies.State('begin_input', 'value'),
               dash.dependencies.State('end_input', 'value'),
               dash.dependencies.State('num_nodes_input', 'value'),
               dash.dependencies.State('nodes_input', 'value')
               ]
              )
def update_graph(n_clicks, function_value, start_value, stop_value, num_value, begin_value, end_value, num_nodes_input, nodes_input):
    X = Symbol('x')
    fun = parse_expr(function_value)

    start = parse_expr(start_value)
    stop = parse_expr(stop_value)

    f = lambdify(X, fun, 'numpy')

    nodes = np.linspace(float(begin_value), float(end_value), int(num_nodes_input))
    steps = np.linspace(float(start), float(stop), int(num_value))
    x = np.append(steps, nodes)
    x.sort()

    data = []
    data.append(go.Scatter(
        x=x,
        y=f(x),
        mode='lines',
        name=str(fun)
    ))

    fun_result = get_result_function(nodes, function_value)
    fr = lambdify(X, fun_result, 'numpy')

    data.append(go.Scatter(
        x=x,
        y=fr(x),
        mode='lines',
        name='Wyniki interpolacji przy pomocy węzłów równoodległych'
    ))

    data.append(go.Scatter(
        x=nodes,
        y=fr(nodes),
        mode='markers',
        name='węzły równoodległe'
    ))

    data = draw_interpolation_with_chebyshev_nodes(
        data,
        function_value,
        float(begin_value),
        float(end_value),
        int(num_nodes_input),
        steps
    )

    data = draw_interpolation_with_own_nodes(
        data,
        function_value,
        steps,
        nodes_input,
    )

    figure = {
        'data': data,
        'layout': go.Layout(
            height=800,
            legend=dict(orientation='h')
        )
    }

    return figure


def draw_interpolation_with_own_nodes(data, function_value, steps, nodes_input):
    x = Symbol('x')
    nodes = np.array([float(num) for num in nodes_input.split(',')])
    fun_result = get_result_function(nodes, function_value)
    fr = lambdify(x, fun_result, 'numpy')
    x_values = np.append(steps, nodes)
    x_values.sort()

    data.append(go.Scatter(
        x=x_values,
        y=fr(x_values),
        mode='lines',
        name='Wyniki interpolacji przy pomocy podanych węzłów'
    ))

    data.append(go.Scatter(
        x=nodes,
        y=fr(nodes),
        mode='markers',
        name='węzły własne'
    ))

    return data


def draw_interpolation_with_chebyshev_nodes(data, function_value, begin_value, end_value, num_nodes, steps):
    x = Symbol('x')
    chebyshev_nodes = create_chebyshev_nodes(begin_value, end_value, num_nodes)
    fun_result = get_result_function(chebyshev_nodes, function_value)
    fr = lambdify(x, fun_result, 'numpy')
    x_values = np.append(steps, chebyshev_nodes)
    x_values.sort()

    data.append(go.Scatter(
        x=x_values,
        y=fr(x_values),
        mode='lines',
        name='Wyniki interpolacji przy pomocy węzłów Czebyszewa'
    ))

    data.append(go.Scatter(
        x=chebyshev_nodes,
        y=fr(chebyshev_nodes),
        mode='markers',
        name='węzły Chebysheva'
    ))

    return data


@app.callback(
              dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('function_input', 'value'),
               dash.dependencies.State('begin_input', 'value'),
               dash.dependencies.State('end_input', 'value'),
               dash.dependencies.State('num_nodes_input', 'value')]
              )
def update_output(n_click, function_value, begin_value, end_value, num_nodes_input):
    nodes = np.linspace(float(begin_value), float(end_value), int(num_nodes_input))
    result = get_result_function(nodes, function_value)
    return str(result).format()


@app.callback(
              dash.dependencies.Output('display-chebyshev-value', 'children'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('function_input', 'value'),
               dash.dependencies.State('begin_input', 'value'),
               dash.dependencies.State('end_input', 'value'),
               dash.dependencies.State('num_nodes_input', 'value')]
              )
def update_output(n_click, function_value, begin_value, end_value, num_nodes_input):
    chebyshev_nodes = create_chebyshev_nodes(begin_value, end_value, num_nodes_input)
    fun_result = get_result_function(chebyshev_nodes, function_value)
    return str(fun_result).format()


@app.callback(
              dash.dependencies.Output('display-own-nodes-value', 'children'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('function_input', 'value'),
               dash.dependencies.State('nodes_input', 'value'),
               ]
              )
def update_output(n_click, function_value, nodes_input):
    nodes = np.array([float(num) for num in nodes_input.split(',')])
    fun_result = get_result_function(nodes, function_value)
    return str(fun_result).format()


if __name__ == '__main__':
    app.run_server(debug=True)
