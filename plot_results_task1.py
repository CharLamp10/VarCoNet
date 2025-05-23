import pickle
import numpy as np
import os
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

losses = []
with open(os.path.join('results', 'losses_tau_0.05_0.005.txt'), 'r') as f:
    for line in f:
        losses.append(float(line.strip()))
losses = np.array(losses)
        
with open(os.path.join('results', 'test_results_tau_0.05_0.005.pkl'), 'rb') as f:
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

for test in test_result:
    mean_acc1.append(test[1])
    std_acc1.append(test[2])
    mean_acc2.append(test[3])
    std_acc2.append(test[4])
    mean_acc3.append(test[7])
    std_acc3.append(test[8])
    mean_acc4.append(test[11])
    std_acc4.append(test[12])

mean_acc1 = np.array(mean_acc1)
std_acc1 = np.array(std_acc1)
mean_acc2 = np.array(mean_acc2)
std_acc2 = np.array(std_acc2)
mean_acc3 = np.array(mean_acc3)
std_acc3 = np.array(std_acc3)
mean_acc4 = np.array(mean_acc4)
std_acc4 = np.array(std_acc4)

mean_acc5 = [None]
std_acc5 = [None]
mean_acc6 = [None]
std_acc6 = [None]

x = np.arange(1, len(test_result)+1)

fig = go.Figure()

line_colors = ['rgba(255,0,0,1)',     # Red
               'rgba(0,0,255,1)',     # Blue
               'rgba(0,255,0,1)',     # Green
               'rgba(255,0,255,1)',   # Magenta
               'rgba(0,255,255,1)',   # Cyan
               'rgba(165,42,42,1)']   # Brown 
               

lengths = [56, 84, 140, 300, 600, 1200]
# Adding all accuracy lines and their shaded areas
for i in range(1, 7): #7
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
    if not None in mean_acc:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),  # x values for the area
            y=np.concatenate([mean_acc + std_acc, (mean_acc - std_acc)[::-1]]),  # Upper and lower bounds
            fill='toself',
            fillcolor=line_color.replace('1)', '0.2)'),  # Fill color matching the line color with transparency
            line=dict(color='rgba(255,255,255,0)'),
            name=f'±1 Std Dev {i}',
            showlegend=False,
        ))

#horizontal_lines = [0.14, 0.20, 0.26, 0.31, 0.37, 0.50]
horizontal_lines = [0.14, 0.20, 0.31, 0.50, 0.61, 0.69]

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
    yaxis=dict(
        tickvals=[0.1,0.2,0.3,0.4,0.5,0.6,0.7],  # Keeping the same tick positions
        ticktext=[str(label) for label in [0.1,0.2,0.3,0.4,0.5,0.6,0.7]],
        range=[0, 0.7]
    ),
    xaxis_title_font=dict(size=20),  # Change x-axis title font size
    yaxis_title_font=dict(size=20),  # Change y-axis title font size
    legend=dict(
        font=dict(size=16)  # Change the legend font size
    )
)
#fig.write_image('subject_fingerprinting_rates_final.png',scale=8, engine = "orca")



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
#fig.write_image('task1_training_loss.png', scale=8, engine = "orca")


accs1_pcc = np.load('pcc_acc_l_56.npy')
accs2_pcc = np.load('pcc_acc_l_84.npy')
accs3_pcc = np.load('pcc_acc_l_140.npy')
accs4_pcc = np.load('pcc_acc_l_300.npy')

test_result = test_result[10][0]
accs1 = test_result[np.arange(0,90,6)]
accs2 = test_result[np.arange(1,90,6)]
accs3 = test_result[np.arange(3,90,6)]
accs4 = test_result[np.arange(5,90,6)]

accs = [accs1,accs2,accs3,accs4]
accs_pcc = [accs1_pcc,accs2_pcc,accs3_pcc,accs4_pcc]

data = []
labels = []
groups = []

group = ['l=56','l=84','l=140','l=300']
for i in range(len(group)):
    data.extend(accs_pcc[i])
    labels.extend(['PCC'] * 15)
    groups.extend([group[i]] * 15)
    
    data.extend(accs[i])
    labels.extend(['VarCoNet'] * 15)
    groups.extend([group[i]] * 15)
    

df = pd.DataFrame({'Value': data, 'Method': labels, 'Group': groups})

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

ax = sns.boxplot(x='Group', y='Value', hue='Method', data=df, palette='Set2')

for i in range(0, len(group)-1, 2):  # Iterate every other group
    ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.5)

plt.ylabel('Accuracy', fontsize=25)
plt.xlabel('')

plt.legend(loc='lower right', fontsize = 20)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig('comparison_boxplot.png', dpi=600, bbox_inches='tight')
plt.show()



data = []
labels = []
groups = []

group = ['T=40 s','T=60 s','T=100 s','T=216 s']
for i in range(len(group)):
    data.extend(accs_pcc[i])
    labels.extend(['PCC'] * 15)
    groups.extend([group[i]] * 15)
    
    data.extend(accs[i])
    labels.extend(['VarCoNet'] * 15)
    groups.extend([group[i]] * 15)
    

df = pd.DataFrame({'Value': data, 'Method': labels, 'Group': groups})

rgb_color = (240/255, 238/255, 231/255)
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor(rgb_color)
sns.set(style="whitegrid")

ax = sns.boxplot(x='Group', y='Value', hue='Method', data=df, palette='Set2')

for i in range(0, len(group)-1, 2):
    ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.5)

plt.ylabel('Accuracy', fontsize=25)
plt.xlabel('')

plt.legend(loc='lower right', fontsize=20)
plt.xticks(rotation=0, fontsize=20)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig('comparison_boxplot_background.png', dpi=600, bbox_inches='tight')
plt.show()