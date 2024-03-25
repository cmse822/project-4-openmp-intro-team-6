import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/data_omp_thread_to_thread_speedup.csv')

# Define colors for the groups
colors = plt.cm.tab10.colors

# Group the data by 'matrix_size'
grouped = df.groupby('matrix_size')

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
for i, (name, group) in enumerate(grouped):
    color = colors[i % len(colors)]  # Cycle through colors
    plt.scatter(group['thread_count'], group['time_to_compute'], label=name, color=color)
    # Connect points with lines
    plt.plot(group['thread_count'], group['time_to_compute'], color=color, linestyle='-', marker='o')

# Adding labels and title
plt.xlabel('Thread Count')
plt.ylabel('Time to Compute')
plt.title('Time to Compute vs. Thread Count Grouped by Matrix Size')

# Adding legend
plt.legend(title='Matrix Size')

# Save the plot as an image file 
plt.savefig('../data/plot_tts.png')

# Displaying the plot
plt.show()