import pandas as pd
import matplotlib.pyplot as plt
import re

# Load the data from Excel file
df = pd.read_excel('AOI_data.xlsx')

# Sort the dataframe by Frame and Contours
df = df.sort_values(by=['Frame', 'Contours'])

# Initialize colors for plotting
colors = ['red', 'blue', 'green']

# Initialize plot
plt.figure(figsize=(8, 6))

# Initialize dictionary to store last three contours for each frame
last_three_contours = {}

# Iterate over rows in dataframe
for idx, row in df.iterrows():
    frame = row['Frame']
    contour_str = row['Contours']
    x = row['X']
    y = row['Y']
    
    # Extract contour number using regular expression
    contour_match = re.match(r'Contour (\d+)', contour_str)
    if contour_match:
        contour = int(contour_match.group(1))
    else:
        contour = None
    
    if frame not in last_three_contours:
        last_three_contours[frame] = []
    
    # Add contour to last three contours list for this frame
    if contour is not None:
        last_three_contours[frame].append((contour, x, y))
        if len(last_three_contours[frame]) > 3:
            last_three_contours[frame].pop(0)

# Iterate over last three contours for each frame and plot them
for frame, contours in last_three_contours.items():
    for i, (contour, x, y) in enumerate(contours):
        color = colors[i]
        plt.plot(x, y, marker='o', color=color, linestyle='-', linewidth=1, label=f'Frame {frame}, Contour {contour}')

# Set plot labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of Coordinates')

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()
