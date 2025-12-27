"""
Graphing and visualization utilities for the Bitcoin Funding Rate Volatility Prediction project.
"""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_funding_rate(
    df: pd.DataFrame,
    title: str = "Funding Rate vs. Time",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the funding rate over time.
    
    Args:
        df: DataFrame containing funding rate data
        title: Plot title
        figsize: Figure size tuple
        save_path: Path to save the plot (optional)
    """
    # Make a copy of the DataFrame and convert the timestamp
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df = df.set_index('timestamp')

    # Plot the funding rate, scaled for better visualization
    plt.figure(figsize=figsize)
    plt.plot(df.index, df['funding_rate'] * 10_000, label='Funding Rate (bips)', color='blue')
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Funding Rate (bips)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()  # Adjust layout to avoid clipping
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    feature_names: list,
    importance_values: list,
    title: str = "Feature Importance",
    figsize: tuple = (10, 6),
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature importance values.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Plot title
        figsize: Figure size tuple
        top_n: Number of top features to display
        save_path: Path to save the plot (optional)
    """
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()