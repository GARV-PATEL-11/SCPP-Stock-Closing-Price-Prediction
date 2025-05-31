import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from plotly.subplots import make_subplots

def learning_rate_schedule_plot(history, model_type, output_dir):
    """
    Desktop learning rate schedule chart (1200x650)
    Shows learning rate changes over epochs with detailed analysis
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = go.Figure()
    
    # Add learning rate trace
    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['learning_rates'],
            mode='lines+markers',
            name='Learning Rate',
            line=dict(color='#f39c12', width=3),
            marker=dict(size=6, color='#f39c12', opacity=0.8),
            hovertemplate='<b>Epoch:</b> %{x}<br><b>Learning Rate:</b> %{y:.8f}<extra></extra>',
        )
    )
    
    # Add trend line if learning rate changes significantly
    lr_values = np.array(history['learning_rates'])
    if np.std(lr_values) > np.mean(lr_values) * 0.1:  # If LR varies significantly
        z = np.polyfit(history['epochs'], lr_values, 2)  # Quadratic fit
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=history['epochs'],
                y=p(history['epochs']),
                mode='lines',
                name='LR Trend',
                line=dict(color='#e67e22', width=2, dash='dot'),
                hoverinfo='skip'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{model_type} Model: Learning Rate Schedule</b>',
            x=0.5,
            font=dict(size=22, color='#2c3e50')
        ),
        width=1200,
        height=650,
        xaxis=dict(
            title='<b>Epoch</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Learning Rate</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False,
            type='log'  # Log scale for better visualization
        ),
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=14)
        ),
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Save file
    filename = os.path.join(output_dir, f"{model_type}_learning_rate.html")
    fig.write_html(filename)
    return filename

    # Generate training history comparison
    history_file = create_training_history_comparison(history, model_type, output_dir)
    files.append(('Training History', history_file))
    
    # Generate learning rate schedule
    lr_file = create_learning_rate_schedule(history, model_type, output_dir)
    files.append(('Learning Rate Schedule', lr_file))
        