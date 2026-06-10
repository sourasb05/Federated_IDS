import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 8))
ax.axis('off')

# Define the coordinates for the nodes based on Rank
nodes = {
    'root': (3, 6),
    '1_1': (2, 4.5), '1_2': (4, 4.5),
    '2_1': (1.5, 3), '2_2': (3, 3), '2_3': (4.5, 3),
    '3_1': (1.25, 1.5), '3_2': (3, 1.5), '3_3': (4.75, 1.5)
}

# Define color mappings for the nodes (graded from black to light gray)
colors = {
    'root': '#000000',
    '1_1': '#555555', '1_2': '#555555',
    '2_1': '#888888', '2_2': '#888888', '2_3': '#888888',
    '3_1': '#BBBBBB', '3_2': '#BBBBBB', '3_3': '#BBBBBB'
}

# Define the directed upward edges (from, to)
edges = [
    ('1_1', 'root'), ('1_2', 'root'),
    ('2_1', '1_1'), ('2_2', '1_1'), ('2_2', '1_2'), ('2_3', '1_2'),
    ('3_1', '2_1'), ('3_2', '2_2'), ('3_3', '2_3')
]

# Draw directional arrows
for start, end in edges:
    x1, y1 = nodes[start]
    x2, y2 = nodes[end]
    
    # Calculate offsets so arrows touch the edge of the circle, not the center
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    
    # Radius of circles is 0.35, shorten arrow reach accordingly
    shorten = 0.35
    ratio = shorten / length
    
    ax.annotate('', 
                xy=(x2 - dx * ratio, y2 - dy * ratio), 
                xytext=(x1 + dx * ratio, y1 + dy * ratio),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

# Draw node circles
for node, (x, y) in nodes.items():
    circle = patches.Circle((x, y), radius=0.35, facecolor=colors[node], zorder=3)
    ax.add_patch(circle)

# Add top root label
plt.text(3, 6.5, 'DODAG root', ha='center', fontsize=12)

# Add Rank text and dashed dividers
y_positions = [6, 4.5, 3, 1.5]
for i, y in enumerate(y_positions):
    if i == 0:
        plt.text(0.5, y + 0.1, 'Rank', ha='center', fontsize=11)
        
    plt.text(0.5, y - 0.2, str(i), ha='center', fontsize=11)
    
    # Draw dashed lines separating ranks (except below the bottom rank)
    if i < 3:
        ax.plot([0.5, 5.5], [y - 0.75, y - 0.75], color='lightgray', linestyle='--', dashes=(3, 3), zorder=1)

# Set rendering bounds
plt.xlim(0, 6)
plt.ylim(0.5, 7)
plt.show()