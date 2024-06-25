import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load your data
df = pd.read_csv('outlier_processing.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

views = ['L', 'R']
positions = ['CC', 'MLO']
readers = ['-1', '-2']
epsilon = 0  # Small constant to avoid log(0)
# outlier_type = 'ssqd'
outlier_type, port = 'median_dist', 8053
# outlier_type, port = 'm_median_dist', 8052
# outlier_type = 'median_sq_dist'

# Calculate the number of outliers for a range of threshold values for both views
def calculate_outliers(view):
    min_th = int(df[[f'{view}_MLO-1_{outlier_type}', f'{view}_MLO-2_{outlier_type}',
                     f'{view}_CC-1_{outlier_type}', f'{view}_CC-2_{outlier_type}']].min().min())
    max_th = int(df[[f'{view}_MLO-1_{outlier_type}', f'{view}_MLO-2_{outlier_type}',
                     f'{view}_CC-1_{outlier_type}', f'{view}_CC-2_{outlier_type}']].max().max())
    threshold_range = list(range(min_th, max_th))
    threshold_ticks = list(range(min_th, max_th, int((max_th-min_th)/15)))
    outliers_per_threshold = []
    for threshold in threshold_range:
        df[f'{view}_is_outlier'] = (
            (df[f'{view}_MLO-1_{outlier_type}'] >= threshold) |
            (df[f'{view}_MLO-2_{outlier_type}'] >= threshold) |
            (df[f'{view}_CC-1_{outlier_type}'] >= threshold) |
            (df[f'{view}_CC-2_{outlier_type}'] >= threshold)
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
min_th = min(df[[f'{view}_MLO-1_{outlier_type}', f'{view}_MLO-2_{outlier_type}',
                    f'{view}_CC-1_{outlier_type}', f'{view}_CC-2_{outlier_type}']].min().min() for view in views)
max_th = max(df[[f'{view}_MLO-1_{outlier_type}', f'{view}_MLO-2_{outlier_type}',
                    f'{view}_CC-1_{outlier_type}', f'{view}_CC-2_{outlier_type}']].max().max() for view in views)
app.layout = html.Div([
    html.H1("Outlier Detection and Visualization"),
    dcc.Graph(id='outlier-histogram'),
    dcc.Graph(id='threshold-plot', figure=static_fig),
    dcc.Slider(
        id='threshold-slider',
        min=min_th,
        max=max_th,
        step=0.1,
        value=1.,
        marks={i: str(i) for i in range(int(min_th),
                                        int(max_th) + 1,
                                        int((max_th - min_th) / 15))},
        updatemode='drag'
    ),
    dcc.Graph(id='scatter-plot-outliers'),
    dcc.Graph(id='scatter-plot-inliers')
])

@app.callback(
    [Output('outlier-histogram', 'figure'),
     Output('threshold-plot', 'figure'),
     Output('scatter-plot-outliers', 'figure'),
     Output('scatter-plot-inliers', 'figure')],
    [Input('threshold-slider', 'value')]
)
def update_plots(threshold):
    # Identify outliers based on the threshold for both views
    separately_filtered = {}
    separately_unfiltered = {}
    for view in views:
        for pos in positions:
            df[f'{view}_{pos}_is_outlier'] = ((df[f'{view}_{pos}-1_{outlier_type}'] >= threshold) |
                                              (df[f'{view}_{pos}-2_{outlier_type}'] >= threshold))
            separately_filtered[f'{view}_{pos}_is_outlier'] = df[df[f'{view}_{pos}_is_outlier'] == True]
            separately_unfiltered[f'{view}_{pos}_is_outlier'] = df[df[f'{view}_{pos}_is_outlier'] == False]
        df[f'{view}_is_outlier'] = (
                (df[f'{view}_MLO-1_{outlier_type}'] >= threshold) |
                (df[f'{view}_MLO-2_{outlier_type}'] >= threshold) |
                (df[f'{view}_CC-1_{outlier_type}'] >= threshold) |
                (df[f'{view}_CC-2_{outlier_type}'] >= threshold)
        )

    # Filter the DataFrame to keep only non-outliers
    df_unfiltered_L = df[df['L_is_outlier'] == False]
    df_unfiltered_R = df[df['R_is_outlier'] == False]
    # Filter the DataFrame to keep only outliers
    df_filtered_L = df[df['L_is_outlier'] == True]
    df_filtered_R = df[df['R_is_outlier'] == True]

    # Create the histogram
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df_filtered_L['VASCombinedAvDensity'],
                                    name='Outliers L', nbinsx=50, opacity=0.5))
    hist_fig.add_trace(go.Histogram(x=df_filtered_R['VASCombinedAvDensity'],
                                    name='Outliers R', nbinsx=50, opacity=0.5))
    hist_fig.update_layout(barmode='overlay', title='Number of Outliers vs VAS Combined Average Density',
                           xaxis_title='VAS Combined Average Density', yaxis_title='Number of Outliers')

    # Get the current number of outliers at the selected threshold for both views
    current_num_outliers_L = df['L_is_outlier'].sum() + epsilon  # Add epsilon to avoid log(0)
    current_num_outliers_R = df['R_is_outlier'].sum() + epsilon  # Add epsilon to avoid log(0)

    # Update the static plot with a vertical line for the current threshold
    threshold_fig = go.Figure()
    threshold_fig.add_trace(go.Scatter(x=threshold_range_L, y=outliers_per_threshold_L,
                                       mode='lines', name='Outliers L'))
    threshold_fig.add_trace(go.Scatter(x=threshold_range_R, y=outliers_per_threshold_R,
                                       mode='lines', name='Outliers R'))
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
                                    # dict(
                                    #     type='line',
                                    #     yref='y', y0=1, y1=current_num_outliers_R,
                                    #     xref='x', x0=threshold, x1=threshold,
                                    #     line=dict(color="Blue", width=2, dash="dash")
                                    # ),
                                    # dict(
                                    #     type='line',
                                    #     yref='y', y0=current_num_outliers_R, y1=current_num_outliers_R,
                                    #     xref='x', x0=epsilon, x1=threshold,
                                    #     line=dict(color="Blue", width=2, dash="dash")
                                    # )
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

    # Create the scatter plot grid for reader comparisons
    scatter_fig_out = make_subplots(rows=1, cols=4, subplot_titles=('LCC_out', 'RCC_out', 'LMLO_out', 'RMLO_out'))
    scatter_fig_in = make_subplots(rows=1, cols=4, subplot_titles=('LCC_in', 'RCC_in', 'LMLO_in', 'RMLO_in'))

    views_full = ['LCC', 'RCC', 'LMLO', 'RMLO']

    for j, pos in enumerate(positions):
        for i, view in enumerate(views):
            row = 1
            col = (j*2) + i + 1
            reader1 = separately_filtered[f'{view}_{pos}_is_outlier'][f'{view}{pos}-1']
            reader1un = separately_unfiltered[f'{view}_{pos}_is_outlier'][f'{view}{pos}-1']
            reader2 = separately_filtered[f'{view}_{pos}_is_outlier'][f'{view}{pos}-2']
            reader2un = separately_unfiltered[f'{view}_{pos}_is_outlier'][f'{view}{pos}-2']
            scatter_fig_out.add_trace(go.Scatter(x=reader1, y=reader2, mode='markers', name=view+'_outliers'),
                                  row=row, col=col)
            scatter_fig_out.add_trace(go.Scatter(x=[min(reader1.min(), reader2.min())],
                                             y=[max(reader1.max(), reader2.max())],
                                             mode='lines', line=dict(color='gray', dash='dash')),
                                  row=row, col=col)
            scatter_fig_out.update_xaxes(title_text='Reader 1', row=row, col=col)
            scatter_fig_out.update_yaxes(title_text='Reader 2', row=row, col=col)
            scatter_fig_in.add_trace(go.Scatter(x=reader1un, y=reader2un, mode='markers', name=view+'_inliers'),
                                  row=row, col=col)
            scatter_fig_in.add_trace(go.Scatter(x=[min(reader1.min(), reader2.min())],
                                             y=[max(reader1.max(), reader2.max())],
                                             mode='lines', line=dict(color='gray', dash='dash')),
                                  row=row, col=col)
            scatter_fig_in.update_xaxes(title_text='Reader 1', row=row, col=col)
            scatter_fig_in.update_yaxes(title_text='Reader 2', row=row, col=col)

    # for i, view in enumerate(views_full):
    #     row = i // 2 + 1
    #     col = i % 2 + 1
    #     if 'L' in view:
    #         reader1 = df_unfiltered_L[f'{view}-1']
    #         reader2 = df_unfiltered_L[f'{view}-2']
    #     if 'R' in view:
    #         reader1 = df_unfiltered_R[f'{view}-1']
    #         reader2 = df_unfiltered_R[f'{view}-2']
    #     scatter_fig.add_trace(go.Scatter(x=reader1, y=reader2, mode='markers', name=view), row=row, col=col)
    #     scatter_fig.add_trace(go.Scatter(x=[min(reader1.min(), reader2.min())], y=[max(reader1.max(), reader2.max())],
    #                                      mode='lines', line=dict(color='gray', dash='dash')), row=row, col=col)
    # for i, view in enumerate(views_full):
    #     row = i // 2 + 1
    #     col = i % 2 + 3
    #     if 'L' in view:
    #         reader1 = df_filtered_L[f'{view}-1']
    #         reader2 = df_filtered_L[f'{view}-2']
    #     if 'R' in view:
    #         reader1 = df_filtered_R[f'{view}-1']
    #         reader2 = df_filtered_R[f'{view}-2']
    #     scatter_fig.add_trace(go.Scatter(x=reader1, y=reader2, mode='markers', name=view), row=row, col=col)
    #     scatter_fig.add_trace(go.Scatter(x=[min(reader1.min(), reader2.min())], y=[max(reader1.max(), reader2.max())],
    #                                      mode='lines', line=dict(color='gray', dash='dash')), row=row, col=col)

    scatter_fig_out.update_layout(height=400, width=1600, title_text='Outlier Reader Comparisons',
                                  showlegend=False)
    scatter_fig_in.update_layout(height=400, width=1600, title_text='Inlier Reader Comparisons',
                                 showlegend=False)

    return hist_fig, threshold_fig, scatter_fig_out, scatter_fig_in

if __name__ == '__main__':
    app.run_server(debug=False, port=port)
