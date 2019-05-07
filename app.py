import os

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = np.sin(x)

trace1 = go.Scatter(
    x=x,
    y=y,
    mode='lines',
)

# use numpy's built in trapezoid-rule integration tool
dy = np.trapz(y, x)

app.layout = html.Div([
    html.H1('Interpolacja funkcji metodą Hermite’a.'),
    html.P('Projekt na WZM'),
    html.Div([
        html.Div([
            html.H5('Funkcja:'),
            dcc.Input(
                value='sin(x)',
                type='text',
                id='function_input'
            ),
            html.H5('Zakres:'),
            html.H6('OD:'),
            dcc.Input(
                value=-1.0,
                type='number',
                id='begin_input'
            ),
            html.H6('DO:'),
            dcc.Input(
                value=1.0,
                type='number',
                id='end_input'
            ),
            html.H5('Typ rozkładu węzłów:'),
            dcc.Dropdown(
                id='nodes_type_dropdown',
                options=[{'label': i, 'value': i} for i in ['równoodległe', 'Czebyszewa', 'własne']],
                value='równoodległe'
            ),
            html.P(''),
            html.Button(
                'Oblicz',
                id='start_button',
            ),
            html.H6('Wynikowy wielomina:'),
            html.Div(
                id='display-value',
                children = 'Enter a value and press submit'
            ),
        ], className="two columns"),
        html.Div([
            html.H6('Wykres:'),
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                    )],
                    'layout': go.Layout(
                        hovermode='closest',
                        height=650
                    )
                }
            )
        ], className="ten columns")
    ], className='row')
])

@app.callback(
              dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('start_button', 'n_clicks')],
              [dash.dependencies.State('begin_input', 'value')]
              )

def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks
    )



if __name__ == '__main__':
    app.run_server(debug=True)