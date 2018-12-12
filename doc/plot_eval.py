import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

# -------- Load data -----------
data_dir = "eval_value_error"
X, Y, names = [], [], []
for file_name in os.listdir(data_dir):
    eval_avg_reward = pd.read_csv(os.path.join(data_dir, file_name))
    x = np.array(eval_avg_reward["Step"])
    y = np.array(eval_avg_reward["Value"])
    if len(x) > 71000 // 200:
        x = x[:71000 // 200]
        y = y[:71000 // 200]
    X.append(x)
    Y.append(y)
    m = re.search("run_CarRacing-v0-(.+?)-tag-*", file_name)
    if m: names.append(m.group(1))
    else: names.append(file_name)

x_max = np.amax([x[-1] for x in X])

# -------- Figure setup -----------
plt.figure(figsize=(10,5))

# Remove the plot frame lines. They are unnecessary chartjunk.    
ax = plt.subplot(111)    
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False)    

# Ensure that the axis ticks only show up on the bottom and left of the plot.    
# Ticks on the right and top of the plot are generally unnecessary chartjunk.    
ax.get_xaxis().tick_bottom()    
ax.get_yaxis().tick_left()  

# Limit the range of the plot to only where the data is.    
# Avoid unnecessary whitespace.
plt.ylim(-100, 1000)
plt.xlim(-100, x_max)

plt.ylabel("Score", fontsize=12)
plt.xlabel("Epochs", fontsize=12)

# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    
plt.yticks(range(-100, 1001, 100), [str(x) for x in range(-100, 1001, 100)], fontsize=12)
plt.xticks(fontsize=12)

# Provide tick lines across the plot to help your viewers trace along    
# the axis ticks. Make sure that the lines are light and small so they    
# don't obscure the primary data lines.
for y in range(-100, 1001, 100):
    plt.plot([-100, x_max], [y, y], "--", lw=0.5, color="black", alpha=0.3)

# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")   

# -------- Plot data -----------

# This function takes an array of numbers and smoothes them out.  
# Smoothing is useful for making plots a little easier to read.  
def sliding_window(data_array, function, window=5):
    data_array = np.array(data_array)  
    new_list = []  
    for i in range(len(data_array)):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
        new_list.append(function(data_array[indices]))
    return np.array(new_list)  

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]   

for i, (x, y, name) in enumerate(zip(X, Y, names)):
    means = sliding_window(y, function=lambda x: np.mean(x))
    stds  = sliding_window(y, function=lambda x: np.std(x))

    last100 = y[-100:]
    last100_mean = np.mean(last100)
    last100_std  = np.std(last100)
    print("name", name,
          "last100_mean", last100_mean,
          "last100_std", last100_std)

    # Use matplotlib's fill_between() call to create error bars.
    plt.fill_between(x, means - stds, means + stds, color=np.array(tableau20[i*2+1]) / 255.0, alpha=0.3)  
    #plt.plot(x, y, color=np.array(tableau20[i]) / 255.0, alpha=0.3)
    plt.plot(x, means, color=np.array(tableau20[i*2]) / 255.0, label=name)

plt.legend(loc=(0.05, 0.75))
plt.savefig("{}.png".format(data_dir), bbox_inches="tight")
plt.show()