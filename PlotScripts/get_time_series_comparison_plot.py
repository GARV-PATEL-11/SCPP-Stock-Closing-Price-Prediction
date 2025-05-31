import os
from datetime import datetime

import numpy as np
import plotly.graph_objects as go


def time_series_comparison_plot(targets_original, predictions_original, model_type, output_dir, phase):
    """
    Generates a time series comparison plot between actual and predicted values.
    - Saves the plot as an HTML file in the specified output directory.
    - Resolution: 1500x750
    """

    timesteps = list(range(len(targets_original)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = go.Figure()

    # Actual values trace
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=targets_original.flatten(),
            mode='lines+markers',
            name='Actual Values',
            line=dict(color='#193142', width=1), 
            marker=dict(size=2, symbol='circle'),
            hovertemplate='<b>Time Step:</b> %{x}<br><b>Actual Price:</b> $%{y:.4f}<extra></extra>',
        )
    )

    # Predicted values trace
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=predictions_original.flatten(),
            mode='lines+markers',
            name='Predicted Values',
            line=dict(color='#CC5500', width=1),
            marker=dict(size=2, symbol='diamond'),
            hovertemplate='<b>Time Step:</b> %{x}<br><b>Predicted Price:</b> $%{y:.4f}<extra></extra>',
        )
    )

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f'<b>{model_type} Model: Time Series Prediction Analysis</b>',
            x=0.5,
            font=dict(size=22, color='#2c3e50')
        ),
        width=1500,
        height=750,
        xaxis=dict(
            title='<b>Time Steps</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Stock Price ($)</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,1)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=90, r=90, t=100, b=90)
    )

    # Save plot
    filename = os.path.join(output_dir, f"{model_type}_{phase}_Actual_vs_Predicted.html")
    fig.write_html(filename)
    return filename
