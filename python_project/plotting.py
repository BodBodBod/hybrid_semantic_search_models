import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import kaleido
from scipy.spatial import Delaunay

def plot_3d_recall(df, file_path):
    models = df['model'].unique()

    for model in models:
        subset = df[df['model'] == model]

        x = subset['alpha (BM25)'].values
        y = subset['beta (TF-IDF)'].values
        z = subset['gamma (Transformer)'].values
        c = subset['recall@5'].values

        # triangulation
        points2D = np.vstack([x, y]).T
        tri = Delaunay(points2D)

        fig = go.Figure()

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            intensity=c,
            colorscale='RdBu',
            reversescale=True,
            colorbar=dict(
                title='Recall@5',
                len=0.7
            ),
            opacity=0.8,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            showscale=True
        ))

        fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>Модель: {model.split('/')[1]}</b>",
                    x=.5,
                    y=.85,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=16),
                    xanchor='center',
                    yanchor='bottom'
                )
            ],
            scene=dict(
                xaxis=dict(title='Alpha (BM25)', range=[0, 1]),
                yaxis=dict(title='Beta (TF-IDF)', range=[0, 1]),
                zaxis=dict(title='Gamma (Transformer)', range=[0, 1]),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.8, z=.8))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=500,
            width=600
        )

        # save as png
        os.makedirs(file_path + 'experiments_results/', exist_ok=True)
        fig.write_image(
            file_path + f"experiments_results/{model.split('/')[1]}.png",
            scale=5  # image quality
        )

        fig.show()
