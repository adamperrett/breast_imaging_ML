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

view = 'R'
epsilon = 0#1e-0  # Small constant to avoid log(0)

# Calculate the number of outliers for a range of threshold values
threshold_range = list(range(int(df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                                     view+'_CC-1_median_dist', view+'_CC-2_median_dist']].min().min()),
                             int(df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                                     view+'_CC-1_median_dist', view+'_CC-2_median_dist']].max().max()) + 1))
threshold_ticks = list(range(int(df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                                     view+'_CC-1_median_dist', view+'_CC-2_median_dist']].min().min()),
                             int(df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                                     view+'_CC-1_median_dist', view+'_CC-2_median_dist']].max().max()) + 1, 10))

outliers_per_threshold = []
for threshold in threshold_range:
    df['is_outlier'] = (
        (df[view+'_MLO-1_median_dist'] > threshold) |
        (df[view+'_MLO-2_median_dist'] > threshold) |
        (df[view+'_CC-1_median_dist'] > threshold) |
        (df[view+'_CC-2_median_dist'] > threshold)
    )
    outliers_per_threshold.append(df['is_outlier'].sum() + epsilon)  # Add epsilon to avoid log(0)

# Create the static plot for the number of outliers per threshold
static_fig = go.Figure()
static_fig.add_trace(go.Scatter(x=threshold_range, y=outliers_per_threshold, mode='lines', name='Outliers'))
static_fig.update_layout(title='Number of Outliers per Threshold',
                         xaxis_title='Threshold',
                         yaxis_title='Number of Outliers',
                         yaxis_type='log')  # Set y-axis to logarithmic scale

app.layout = html.Div([
    html.H1("Outlier Detection and Visualization"),
    dcc.Graph(id='outlier-histogram'),
    dcc.Slider(
        id='threshold-slider',
        min=df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                view+'_CC-1_median_dist', view+'_CC-2_median_dist']].min().min(),
        max=df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                view+'_CC-1_median_dist', view+'_CC-2_median_dist']].max().max(),
        step=0.1,
        value=1.,
        marks={i: str(i) for i in range(int(
            df[[view+'_MLO-1_median_dist', view+'_MLO-2_median_dist',
                view+'_CC-1_median_dist', view+'_CC-2_median_dist']].min().min()),
                                        int(df[[view+'_MLO-1_median_dist',
                                                view+'_MLO-2_median_dist',
                                                view+'_CC-1_median_dist',
                                                view+'_CC-2_median_dist']].max().max()) + 1, 5)},
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
    # Identify outliers based on the threshold
    df['is_outlier'] = (
            (df[view+'_MLO-1_median_dist'] > threshold) |
            (df[view+'_MLO-2_median_dist'] > threshold) |
            (df[view+'_CC-1_median_dist'] > threshold) |
            (df[view+'_CC-2_median_dist'] > threshold)
    )

    # Count the number of outliers
    outlier_counts = df.groupby('VASCombinedAvDensity')['is_outlier'].sum().reset_index()

    # Create the histogram
    hist_fig = px.histogram(outlier_counts, x='VASCombinedAvDensity', y='is_outlier', nbins=50,
                            labels={'is_outlier': 'Number of Outliers',
                                    'VASCombinedAvDensity': 'VAS Combined Average Density'},
                            title='Number of Outliers vs VAS Combined Average Density')

    # Get the current number of outliers at the selected threshold
    current_num_outliers = df['is_outlier'].sum() + epsilon  # Add epsilon to avoid log(0)

    # Update the static plot with a vertical line for the current threshold
    threshold_fig = go.Figure()
    threshold_fig.add_trace(go.Scatter(x=threshold_range, y=outliers_per_threshold, mode='lines', name='Outliers'))
    threshold_fig.add_trace(go.Scatter(x=[threshold], y=[current_num_outliers],
                                       mode='markers', marker=dict(color='red', size=10),
                                       name='Current Threshold'))
    threshold_fig.update_layout(title='Number of Outliers per Threshold',
                                xaxis_title='Threshold',
                                yaxis_title='Number of Outliers',
                                yaxis_type='log',  # Set y-axis to logarithmic scale
                                shapes=[
                                    dict(
                                        type='line',
                                        yref='y', y0=1, y1=current_num_outliers,
                                        xref='x', x0=threshold, x1=threshold,
                                        line=dict(color="Red", width=2, dash="dash")
                                    ),
                                    dict(
                                        type='line',
                                        yref='y', y0=current_num_outliers, y1=current_num_outliers,
                                        xref='x', x0=epsilon, x1=threshold,
                                        line=dict(color="Red", width=2, dash="dash")
                                    )
                                ],
                                xaxis=dict(
                                    tickvals=threshold_ticks + [threshold],
                                    ticktext=[str(i) for i in threshold_ticks] + [f'<b>{threshold}</b>'],
                                    tickmode='array'
                                ),
                                yaxis=dict(
                                    tickvals=[10**i for i in range(0, int(current_num_outliers).bit_length() + 1)] + [current_num_outliers],
                                    ticktext=[str(10**i) for i in range(0, int(current_num_outliers).bit_length() + 1)] + [f'<b>{current_num_outliers}</b>'],
                                    tickmode='array'
                                ))

    return hist_fig, threshold_fig

if __name__ == '__main__':
    app.run_server(debug=False)
