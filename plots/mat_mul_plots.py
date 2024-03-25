import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('../data/part3_matrix_mult.csv')

save_dir = '../data/'

# Group the data by 'MatrixSize'
grouped = df.groupby('MatrixSize')

# Iterate over each group of 'MatrixSize'
for name, group in grouped:
    # Create a new plot for each group
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    
    # Group the data within each 'MatrixSize' group by 'NumRanks'
    ranks_grouped = group.groupby('NumRanks')
    
    # Plotting scatter plots and lines for 'NumRanks' vs 'AvgTime' within each 'MatrixSize' group
    num_ranks_count = len(ranks_grouped)
    colors = plt.cm.tab10.colors  # Get colors from the colormap
    for i, (num_ranks, ranks_group) in enumerate(ranks_grouped):
        # Use a unique color for each 'NumRanks' group
        color = colors[i % len(colors)] if i < len(colors) else colors[i % num_ranks_count]
        plt.scatter(ranks_group['NumThreadsPerRank'], ranks_group['AvgTime'], label=f'NumRanks {num_ranks}', color=color)
        plt.plot(ranks_group['NumThreadsPerRank'], ranks_group['AvgTime'], linestyle='-', marker='o', color=color)
    
    # Adding labels and title
    plt.xlabel('NumThreadsPerRank')
    plt.ylabel('AvgTime')
    plt.title(f'AvgTime vs. NumThreadsPerRank for MatrixSize {name}')
    
    # Adding legend
    plt.legend()
    
    # Save the plot as an image file
    save_path = os.path.join(save_dir, f'plot_matrix_size_{name}.png')
    plt.savefig(save_path)
    
    # Displaying the plot
    plt.show()