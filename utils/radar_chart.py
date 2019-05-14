# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from scipy import interpolate
import numpy as np

if __name__ == "__main__":
    global df
    # Set data
    df = pd.DataFrame({
        'options': ['1', '2', '4', '8'],
        '10 SPs': [38, 1.5, 30, 4],
        '20 SPs': [29, 10, 9, 34],
        '40 SPs': [8, 39, 23, 24],
        '80 SPs': [7, 31, 33, 14],
        '160 SPs': [28, 15, 32, 14]
    })

def make_radar_chart(df):
    # number of variable (Service providers)
    categories = list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(10,10), nrows=1, ncols=1, subplot_kw=dict(projection="polar"))

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories, color='grey', size=8)

    # Draw ylabels
    #ax.set_rlabel_position(0)
    #plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    #plt.ylim(0, 40)

    for i in range(len(df)):
        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values = df.loc[i].drop('options').values.flatten().tolist()
        # Plot data
        #ax.plot(angles, values, linewidth=1, linestyle='solid', label="%s options" % df["options"][i])
        tck, u = interpolate.splprep([angles,values], s=10000)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 1000), tck, der=0)
        ax.plot(xnew, ynew, linewidth=1, linestyle='solid', label="%s options" % df["options"][i])
        #break
    ax.legend()
    #plt.show()
    fig.savefig("results/radar_plot.png")


if __name__ == "__main__":
    make_radar_chart(df)
