import pickle
import numpy as np
import os
import plotly.graph_objects as go

losses = []
with open(os.path.join('results', 'losses.txt'), 'r') as f:
    for line in f:
        losses.append(float(line.strip()))
losses = np.array(losses)
        
with open(os.path.join('results', 'test_results.pkl'), 'rb') as f:
    test_result = pickle.load(f)

test_result = test_result[:-1]
mean_acc1 = []
std_acc1 = []
mean_acc2 = []
std_acc2 = []
mean_acc3 = []
std_acc3 = []
mean_acc4 = []
std_acc4 = []
mean_acc5 = []
std_acc5 = []
for test in test_result:
    mean_acc1.append(test[1])
    std_acc1.append(test[2])
    mean_acc2.append(test[3])
    std_acc2.append(test[4])
    mean_acc3.append(test[5])
    std_acc3.append(test[6])
    mean_acc4.append(test[7])
    std_acc4.append(test[8])
    mean_acc5.append(test[9])
    std_acc5.append(test[10])
    
mean_acc1 = np.array(mean_acc1)
std_acc1 = np.array(std_acc1)
mean_acc2 = np.array(mean_acc2)
std_acc2 = np.array(std_acc2)
mean_acc3 = np.array(mean_acc3)
std_acc3 = np.array(std_acc3)
mean_acc4 = np.array(mean_acc4)
std_acc4 = np.array(std_acc4)
mean_acc5 = np.array(mean_acc5)
std_acc5 = np.array(std_acc5)

x = np.arange(1, len(test_result)+1)

fig = go.Figure()

line_colors = ['rgba(255,0,0,1)',     # Red
               'rgba(0,0,255,1)',     # Blue
               'rgba(0,255,0,1)',     # Green
               'rgba(255,0,255,1)',   # Magenta
               'rgba(0,255,255,1)']    # Cyan

lengths = [56, 84, 112, 140, 180]
# Adding all accuracy lines and their shaded areas
for i in range(1, 6):
    mean_acc = eval(f'mean_acc{i}')
    std_acc = eval(f'std_acc{i}')
    line_color = line_colors[i - 1]
    
    # Adding the mean accuracy line
    fig.add_trace(go.Scatter(
        x=x, y=mean_acc,
        mode='lines',
        name=f'l = {lengths[i-1]}',
        line=dict(width=4, color=line_color),  # Thicker line for mean
    ))

    # Adding the shaded area for standard deviation
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),  # x values for the area
        y=np.concatenate([mean_acc + std_acc, (mean_acc - std_acc)[::-1]]),  # Upper and lower bounds
        fill='toself',
        fillcolor=line_color.replace('1)', '0.2)'),  # Fill color matching the line color with transparency
        line=dict(color='rgba(255,255,255,0)'),
        name=f'±1 Std Dev {i}',
        showlegend=False,
    ))

horizontal_lines = [0.14, 0.20, 0.26, 0.31, 0.37]

for i, y in enumerate(horizontal_lines):
    color = line_colors[i]  # Get the corresponding color for the line
    fig.add_shape(
        type='line',
        x0=x[0], x1=x[-1],  # Extend the line across the x-axis
        y0=y, y1=y,
        line=dict(color=color, width=1.5, dash = 'dash')  # Set the line color and width
    )

custom_xticklabels = [1, 5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# Update layout
fig.update_layout(
    #title='Mean Accuracy with Standard Deviation',
    xaxis_title='Epochs',
    yaxis_title='Accuracy',
    template='plotly_white',
    font=dict(size=16),
    xaxis=dict(
        tickvals=x,  # Keeping the same tick positions
        ticktext=[str(label) for label in custom_xticklabels],  # Custom tick labels
    ),
    xaxis_title_font=dict(size=20),  # Change x-axis title font size
    yaxis_title_font=dict(size=20),  # Change y-axis title font size
    legend=dict(
        font=dict(size=16)  # Change the legend font size
    )
)
fig.write_image('subject_fingerprinting_rates.png',scale=8, engine = "orca")



fig = go.Figure()

# Add the loss curve to the figure
fig.add_trace(go.Scatter(
    x=np.arange(1, len(losses) + 1),  # Epoch numbers (1, 2, 3, ..., N)
    y=losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='red', width=3),  # Customize the line color and width
))

# Customize the layout
fig.update_layout(
    xaxis_title='Epoch',
    yaxis_title='Loss',
    font=dict(size=16),
    xaxis=dict(tickmode='linear', tick0=0, dtick=20),  # X-axis ticks customization
    yaxis=dict(showgrid=True),
)

# Save the figure as a high-resolution image (optional)
fig.write_image('task1_training_loss.png', scale=8, engine = "orca")