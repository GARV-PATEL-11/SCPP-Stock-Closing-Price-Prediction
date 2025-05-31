import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import os
from plotly.subplots import make_subplots

def scatter_plot(targets_original, predictions_original, model_type, output_dir,phase):
    """
    Desktop scatter plot analysis (1200x800)
    Shows actual vs predicted correlation with statistical insights
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate R-squared and other metrics
    correlation_matrix = np.corrcoef(targets_original.flatten(), predictions_original.flatten())
    r_squared = correlation_matrix[0, 1] ** 2
    
    fig = go.Figure()
    
    # Add scatter points with color gradient
    fig.add_trace(
        go.Scatter(
            x=targets_original.flatten(),
            y=predictions_original.flatten(),
            mode='markers',
            name='Predictions',
            marker=dict(
                color=targets_original.flatten(),
                colorscale='Plasma',
                size=8,
                opacity=0.8,
                line=dict(width=1, color='white'),
                colorbar=dict(
                    title="<b>Actual Price ($)</b>", 
                    titleside="right",
                    titlefont=dict(size=14)
                )
            ),
            hovertemplate='<b>Actual:</b> $%{x:.4f}<br><b>Predicted:</b> $%{y:.4f}<extra></extra>',
        )
    )
    
    # Add perfect prediction line
    min_val = min(targets_original.min(), predictions_original.min())
    max_val = max(targets_original.max(), predictions_original.max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction Line',
            line=dict(color='#e74c3c', width=3, dash='dash'),
            hoverinfo='skip'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{model_type} Model: Prediction Accuracy Analysis</b><br><sub>R² = {r_squared:.4f}</sub>',
            x=0.5,
            font=dict(size=22, color='#2c3e50')
        ),
        width=1200,
        height=800,
        xaxis=dict(
            title='<b>Actual Price ($)</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Predicted Price ($)</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=14),
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=80, r=120, t=100, b=80)
    )
    
    # Save file
    filename = os.path.join(output_dir, f"{model_type}_{phase}_scatter_plot.html")
    fig.write_html(filename)
    return filename