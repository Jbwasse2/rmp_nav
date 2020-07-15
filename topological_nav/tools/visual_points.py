import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pudb
from tqdm import tqdm


def get_video_trajectory():
    # cap = cv2.VideoCapture("20200709_125258.mp4")
    #    cap = cv2.VideoCapture("20200709_142115.mp4")
    cap = cv2.VideoCapture("20200710_093518.mp4")
    #    cap = cv2.VideoCapture("20200710_145453.mp4")

    # Check if camera opened successfully
    if cap.isOpened() == False:
        IOError("Error opening video stream or file")

    # Read until video is completed
    trajectory = []
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            # Display the resulting frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            trajectory.append(frame)
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    return np.asarray(trajectory)


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(1920 / 100, 1080 / 100)
points = np.load("points.npy")
reachability = np.load("reach.npy")
trajectory = get_video_trajectory()
matplotlib.use("TkAgg")
counter = 0
for i in tqdm(range(0, trajectory.shape[0], 1)):
    frame = trajectory[i, :, :, :]
    point = points[i, :]
    text = str(point)
    plt.arrow(
        960,
        540,
        -point[1] * 255,
        -point[0] * 255,
        color="r",
        head_width=0.05,
        head_length=0.03,
        linewidth=4,
        length_includes_head=True,
    )
    plt.text(4, 1, text, size=25, ha="left", wrap=True)
    pu.db
    r = "{0:.4f}".format(reachability[i])
    plt.text(1400, 1, r, size=25, ha="right", wrap=True)
    plt.imshow(frame)
    plt.savefig("results/imgs/frame" + f"{counter:04}.png", dpi=100)
    plt.cla()
    plt.clf()
    counter += 1
