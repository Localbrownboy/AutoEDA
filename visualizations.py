import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

def plot_scatter(series1: pd.Series, series2: pd.Series, filename: str) -> None:
    """
    Creates a scatter plot for two pandas Series and saves it to the specified filename.
    
    Parameters:
        series1: pandas Series for the x-axis.
        series2: pandas Series for the y-axis.
        filename: str, the filepath where the plot should be saved.
    """
    x_label = series1.name if series1.name is not None else "Series 1"
    y_label = series2.name if series2.name is not None else "Series 2"
    
    plt.figure(figsize=(8, 6))
    plt.scatter(series1, series2, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Scatter Plot between {x_label} and {y_label}")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def make_scatters(
    df: pd.DataFrame, 
    target_vars: Optional[List[str]] = None, 
    other_vars: Optional[List[str]] = None
) -> None:
    """
    Creates scatter plots for each combination of target and other variables from the DataFrame.
    Each plot is saved in the 'visualizations' folder with a descriptive filename.
    
    Parameters:
        df: pandas DataFrame containing the data.
        target_vars: List of column names to be used as target variables. 
                     If None, defaults to all columns in the DataFrame.
        other_vars: List of column names to be used as other variables. 
                    If None, defaults to all columns in the DataFrame.
    """
    # Default to every column if either parameter is None.
    if target_vars is None:
        target_vars = list(df.columns)
    if other_vars is None:
        other_vars = list(df.columns)
        
    # Create visualizations folder if it doesn't exist.
    os.makedirs("visualizations", exist_ok=True)
    
    # Loop through each combination of target and other variables.
    for target in target_vars:
        for other in other_vars:
            filename = f"visualizations/scatter_{target}_vs_{other}.png"
            plot_scatter(df[target], df[other], filename)
            print(f"Saved scatter plot: {filename}")

if __name__ == "__main__":
    # Example DataFrame with multiple columns.
    data = {
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 3, 5, 7, 11],
        'col3': [1, 4, 9, 16, 25]
    }
    df = pd.DataFrame(data)
    
    # If None is passed, it defaults to every column in the DataFrame.
    make_scatters(df, target_vars=None, other_vars=None)
    print("done visualizing")
