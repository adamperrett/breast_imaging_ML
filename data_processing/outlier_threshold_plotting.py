import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load your data
df = pd.read_csv('outlier_processing.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

views = ['L', 'R']
epsilon = 0  # Small constant to avoid log(0)

# Calculate the number of outliers for a range of threshold values for both views
def calculate_outliers(view):
    threshold_range = list(range(int(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                                         f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].min().min()),
                                 int(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                                         f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].max().max()) + 1))
    threshold_ticks = list(range(int(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                                         f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].min().min()),
                                 int(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                                         f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].max().max()) + 1,
                                 10))
    outliers_per_threshold = []
    for threshold in threshold_range:
        df[f'{view}_is_outlier'] = (
            (df[f'{view}_MLO-1_median_dist'] > threshold) |
            (df[f'{view}_MLO-2_median_dist'] > threshold) |
            (df[f'{view}_CC-1_median_dist'] > threshold) |
            (df[f'{view}_CC-2_median_dist'] > threshold)
        )
        outliers_per_threshold.append(df[f'{view}_is_outlier'].sum() + epsilon)  # Add epsilon to avoid log(0)
    return threshold_range, outliers_per_threshold, threshold_ticks

# Calculate outliers for both views
threshold_range_L, outliers_per_threshold_L, threshold_ticks_L = calculate_outliers('L')
threshold_range_R, outliers_per_threshold_R, threshold_ticks_R = calculate_outliers('R')

# Create the static plot for the number of outliers per threshold for both views
static_fig = go.Figure()
static_fig.add_trace(go.Scatter(x=threshold_range_L, y=outliers_per_threshold_L, mode='lines', name='Outliers L'))
static_fig.add_trace(go.Scatter(x=threshold_range_R, y=outliers_per_threshold_R, mode='lines', name='Outliers R'))
static_fig.update_layout(title='Number of Outliers per Threshold',
                         xaxis_title='Threshold',
                         yaxis_title='Number of Outliers',
                         yaxis_type='log')  # Set y-axis to logarithmic scale

app.layout = html.Div([
    html.H1("Outlier Detection and Visualization"),
    dcc.Graph(id='outlier-histogram'),
    dcc.Slider(
        id='threshold-slider',
        min=min(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                    f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].min().min() for view in views),
        max=max(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                    f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].max().max() for view in views),
        step=0.1,
        value=1.,
        marks={i: str(i) for i in range(int(
            min(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                    f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].min().min() for view in views)),
                                        int(max(df[[f'{view}_MLO-1_median_dist', f'{view}_MLO-2_median_dist',
                                                    f'{view}_CC-1_median_dist', f'{view}_CC-2_median_dist']].max().max() for view in views)) + 1, 5)},
        updatemode='drag'
    ),
    dcc.Graph(id='threshold-plot', figure=static_fig),
])

@app.callback(
    [Output('outlier-histogram', 'figure'),
     Output('threshold-plot', 'figure')],
    [Input('threshold-slider', 'value')]
)
def update_plots(threshold):
    # Identify outliers based on the threshold for both views
    for view in views:
        df[f'{view}_is_outlier'] = (
                (df[f'{view}_MLO-1_median_dist'] > threshold) |
                (df[f'{view}_MLO-2_median_dist'] > threshold) |
                (df[f'{view}_CC-1_median_dist'] > threshold) |
                (df[f'{view}_CC-2_median_dist'] > threshold)
        )

    # Filter the DataFrame to keep only outliers
    df_outliers_L = df[df['L_is_outlier']]
    df_outliers_R = df[df['R_is_outlier']]

    # Create the histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df_outliers_L['VASCombinedAvDensity'],
                                    name='Outliers L', nbinsx=50, opacity=0.5))
    hist_fig.add_trace(go.Histogram(x=df_outliers_R['VASCombinedAvDensity'],
                                    name='Outliers R', nbinsx=50, opacity=0.5))
    hist_fig.update_layout(barmode='overlay', title='Number of Outliers vs VAS Combined Average Density',
                           xaxis_title='VAS Combined Average Density', yaxis_title='Number of Outliers')

    # Get the current number of outliers at the selected threshold for both views
    current_num_outliers_L = df['L_is_outlier'].sum() + epsilon  # Add epsilon to avoid log(0)
    current_num_outliers_R = df['R_is_outlier'].sum() + epsilon  # Add epsilon to avoid log(0)

    # Update the static plot with a vertical line for the current threshold
    threshold_fig = go.Figure()
    threshold_fig.add_trace(go.Scatter(x=threshold_range_L, y=outliers_per_threshold_L, mode='lines', name='Outliers L'))
    threshold_fig.add_trace(go.Scatter(x=threshold_range_R, y=outliers_per_threshold_R, mode='lines', name='Outliers R'))
    threshold_fig.add_trace(go.Scatter(x=[threshold], y=[current_num_outliers_L],
                                       mode='markers', marker=dict(color='red', size=10),
                                       name='Current Threshold L'))
    threshold_fig.add_trace(go.Scatter(x=[threshold], y=[current_num_outliers_R],
                                       mode='markers', marker=dict(color='blue', size=10),
                                       name='Current Threshold R'))
    threshold_fig.update_layout(title='Number of Outliers per Threshold',
                                xaxis_title='Threshold',
                                yaxis_title='Number of Outliers',
                                yaxis_type='log',  # Set y-axis to logarithmic scale
                                shapes=[
                                    dict(
                                        type='line',
                                        yref='y', y0=1, y1=current_num_outliers_L,
                                        xref='x', x0=threshold, x1=threshold,
                                        line=dict(color="Red", width=2, dash="dash")
                                    ),
                                    dict(
                                        type='line',
                                        yref='y', y0=current_num_outliers_L, y1=current_num_outliers_L,
                                        xref='x', x0=epsilon, x1=threshold,
                                        line=dict(color="Red", width=2, dash="dash")
                                    ),
                                    dict(
                                        type='line',
                                        yref='y', y0=1, y1=current_num_outliers_R,
                                        xref='x', x0=threshold, x1=threshold,
                                        line=dict(color="Blue", width=2, dash="dash")
                                    ),
                                    dict(
                                        type='line',
                                        yref='y', y0=current_num_outliers_R, y1=current_num_outliers_R,
                                        xref='x', x0=epsilon, x1=threshold,
                                        line=dict(color="Blue", width=2, dash="dash")
                                    )
                                ],
                                xaxis=dict(
                                    tickvals=list(sorted(set(threshold_ticks_L + threshold_ticks_R + [threshold]))),
                                    ticktext=[str(i) for i in sorted(set(threshold_ticks_L + threshold_ticks_R + [threshold]))],
                                    tickmode='array'
                                ),
                                yaxis=dict(
                                    tickvals=[10**i for i in range(0, int(max(current_num_outliers_L, current_num_outliers_R)).bit_length() + 1)] + [current_num_outliers_L, current_num_outliers_R],
                                    ticktext=[str(10**i) for i in range(0, int(max(current_num_outliers_L, current_num_outliers_R)).bit_length() + 1)] + [f'<b>{current_num_outliers_L}</b>', f'<b>{current_num_outliers_R}</b>'],
                                    tickmode='array'
                                ))

    return hist_fig, threshold_fig

if __name__ == '__main__':
    app.run_server(debug=False)
