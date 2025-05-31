import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from plotly.subplots import make_subplots

def training_history_plot(history, model_type, output_dir):
    """
    Desktop training history comparison chart (1400x700)
    Shows training vs validation loss over epochs with best epoch highlights
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = go.Figure()
    
    # Add training loss trace
    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['train_loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#3498db', width=3),
            hovertemplate='<b>Epoch:</b> %{x}<br><b>Training Loss:</b> %{y:.8f}<extra></extra>',
        )
    )
    
    # Add validation loss trace
    fig.add_trace(
        go.Scatter(
            x=history['epochs'],
            y=history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='<b>Epoch:</b> %{x}<br><b>Validation Loss:</b> %{y:.8f}<extra></extra>',
        )
    )
    
    # Highlight best epochs
    best_epochs = history['best_epoch_indices']
    best_val_losses = history['best_val_losses']
    best_train_losses = [history['train_loss'][history['epochs'].index(epoch)] for epoch in best_epochs]
    
    # Best training loss points
    fig.add_trace(
        go.Scatter(
            x=best_epochs,
            y=best_train_losses,
            mode='markers',
            name='Best Training Loss',
            marker=dict(color='#27ae60', size=12, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate='<b>Best Training Epoch:</b> %{x}<br><b>Training Loss:</b> %{y:.8f}<extra></extra>',
        )
    )
    
    # Best validation loss points
    fig.add_trace(
        go.Scatter(
            x=best_epochs,
            y=best_val_losses,
            mode='markers',
            name='Best Validation Loss',
            marker=dict(color='#9b59b6', size=14, symbol='star', line=dict(width=2, color='white')),
            hovertemplate='<b>Best Validation Epoch:</b> %{x}<br><b>Validation Loss:</b> %{y:.8f}<extra></extra>',
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'<b>{model_type} Model: Training History Analysis</b>',
            x=0.5,
            font=dict(size=22, color='#2c3e50')
        ),
        width=1500,
        height=750,
        xaxis=dict(
            title='<b>Epoch</b>',
            titlefont=dict(size=16),
            tickfont=dict(size=13),
            gridcolor='#ecf0f1',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='<b>Loss</b>',
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
            bgcolor="rgba(255,255,255,0.8)"
        ),
        plot_bgcolor='white',
        paper_bgcolor='#f8f9fa',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Save file
    filename = os.path.join(output_dir, f"{model_type}_training_history.html")
    fig.write_html(filename)
    return filename