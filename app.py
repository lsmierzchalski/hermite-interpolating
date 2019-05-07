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
                value='sin(x)',
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
            html.H6('Typ rozkładu węzłów'),
            dcc.Dropdown(
                id='nodes_type_dropdown',
                options=[{'label': i, 'value': i} for i in ['równoodległe', 'Czebyszewa', 'własne']],
                value='równoodległe'
            ),
            html.Label('liczba wezlow(równoodległe, Czebyszewa):'),
            dcc.Input(
                value=10,
                type='number',
                id='num_nodes_input'
            ),
            html.Label('własne węzły:'),
            dcc.Input(
                value='[-1, 0, 1]',
                type='text',
                id='nodes_input'
            ),
            html.P(''),
            html.Button(
                'Oblicz',
                id='start_button',
            ),
            html.H6('Wynikowy wielomina:'),
            html.Div(
                id='display-value',
                children = 'xd(x)'
            ),
            html.H6('Ustawienia programu:'),
            html.Label('Początek wykresu:'),
            dcc.Input(
                value='-2 * pi',
                type='text',
                id='start_input'
            ),
            html.Label('Koniec wykresu:'),
            dcc.Input(
                value='2 * pi',
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
    result = hermite.hermite_interpolationg(nodes, fun, x)
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
               dash.dependencies.State('num_nodes_input', 'value')
               ]
              )
def update_graph(n_clicks, function_value, start_value, stop_value, num_value, begin_value, end_value, num_nodes_input):
    X = Symbol('x')
    fun = parse_expr(function_value)

    start = parse_expr(start_value)
    stop = parse_expr(stop_value)

    f = lambdify(X, fun, 'numpy')

    #TO DO nodes
    nodes = np.linspace(float(begin_value), float(end_value), int(num_nodes_input))

    steps = np.linspace(float(start), float(stop), int(num_value))
    x = np.append(steps,nodes)
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
        name=str(fun_result)
    ))

    data.append(go.Scatter(
        x=nodes,
        y=fr(nodes),
        mode='markers',
        name='węzły'
    ))

    figure = {
        'data': data,
        'layout': go.Layout(
            height=800
        )
    }

    return figure

@app.callback(
              dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('function_input', 'value'),
               dash.dependencies.State('begin_input', 'value'),
               dash.dependencies.State('end_input', 'value'),
               dash.dependencies.State('num_nodes_input', 'value')]
              )
def update_output(n_click, function_value, begin_value, end_value, num_nodes_input):
    # TO DO nodes
    nodes = np.linspace(float(begin_value), float(end_value), int(num_nodes_input))
    result = get_result_function(nodes, function_value)
    return str(result).format()


if __name__ == '__main__':
    app.run_server(debug=True)