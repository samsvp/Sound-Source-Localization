import json
import numpy as np
import matplotlib.pyplot as plt


with open("Data/data.json") as f:
    data = json.load(f)

with open("Data/data_ipqm.json") as f:
    data_ipqm = json.load(f)

# azimuth(rad), elevation(rad), frequency(kHz)
gab_030719 = {
    "2": (90, 90, 25),"3": (120, 90, 25),"4": (150, 90, 25),"6": (60, 90),"7": (30, 90, 25),
    "8": (30, 90),"9": (60, 90),"10": (90, 90),"11": (120, 90),"12": (150, 90),
    "13": (150, 120),"14": (120, 120),"15": (90, 120),"16": (60, 120),"17": (30, 120),
    "18": (30, 120, 25), "19": (60, 120, 25), "20": (90, 120, 25), "21": (120, 120, 25), "22": (150, 120, 25),
}

gab_110118 = {
    "1": (90, 90, 25),"2": (120, 90, 25),"3": (135, 90, 25),"4": (150, 90),"5": (165, 90, 25),
    "6": (60, 90),"7": (45, 90),"8": (30, 90),"9": (15, 90),"10": (15, 90),
    "11": (180, 90), "12": (90, 80), "13": (90,60), "14": (90, 45), "16": (135, 45), "17": (150, 45),
    "18": (60, 60, 25), "19": (45, 60, 25)
}

def plot_error(values, labels, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width/2, np.abs(values[0]) + 1, width, label="f")
    ax.bar(x + width/2, np.abs(values[1]) + 1, width, label='s')
    ax.set_ylabel('Abs Error + 1')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

def choose_coord(ipqm: bool) -> np.ndarray:
    if ipqm:
        distance = 3 * 10**-2 # Distance between hydrophones in m
        coord = np.array((
                [0,0,distance],
                [0,0,0],
                [0,0,-distance],
                [-distance,0,0]
            ))
    else:
        distance_x = (19.051e-3)/2  # Distance between hydrophones in m
        distance_y = (18.37e-3)/2
        coord = np.array(([-distance_x, -8.41e-3, -distance_y],
                        [distance_x, 0, -distance_y],
                        [distance_x, -8.64e-3, distance_y],
                        [-distance_x, -0.07e-3, distance_y]
                    ))
    return coord