import json
import os
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.animation as animation


def visualize(id, meta):
    frames = np.load("data/frames/" + id + "_frames.npy")
    times = np.load("data/frames/" + id + "_times.npy")
    forces = np.load("data/forces/" + id + ".npy")
    path =  ["processed_data/visualization/", str(id), ".gif"]
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')
    # out = cv.VideoWriter("".join(path) , fourcc, 20, (320, 240), isColor = True)

    epsilon = 1400
    neutral = frames[int(frames.shape[0] * .05)]
    dis = lambda x: np.linalg.norm(x - neutral)
    distances = np.array(list(map(dis,frames)))

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    fig.set_size_inches(11, 9)
    fig.tight_layout()


    def animate(frame_num):
        timestamp = times[frame_num]

        ax0.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax1.set_xlim(min(times[0], forces[0, 0]), max(times[-1], forces[-1, 0]))
        ax3.set_xlim(min(times[0], forces[0, 0]), max(times[-1], forces[-1, 0]))

        ax0.set_title("Frame " + str(frame_num))
        ax0.imshow(frames[frame_num, :, :, 0])

        ax1.set_title("Measured force")
        ax1.vlines(timestamp, np.min(forces[:, 1]), np.max(forces[:, 1]), color = "gray")
        ax1.plot(forces[:, 0], forces[:, 1], color = "#d3d3d3")
        forcesSoFar = forces[:, 0] < timestamp
        ax1.plot(forces[forcesSoFar][:, 0], forces[forcesSoFar][:, 1], color = "blue")

        # ax2.tick_params(left = False, right = False , labelleft = False ,
        #         labelbottom = False, bottom = False)
        ax2.text(0.4, 0.6, meta["object"])
        ax2.text(0.4, 0.5, "Max. force " + str(meta["max_force"]))
        ax2.text(0.4, 0.4, "Max. velocity " + str(meta["max_vel"]))
        
        ax3.set_title("Frame intensity difference to 0th frame")
        ax3.vlines(timestamp, np.min(distances), np.max(distances), color = "gray")
        ax3.plot(times, distances, color = "#d3d3d3")
        ax3.plot(times[: frame_num], distances[: frame_num], color = "blue")
        ax3.hlines(epsilon, times[0], times[-1], colors = ["red"])

        # ax3.imshow(frames[frame_num, :, :, 0])
    

    ani = animation.FuncAnimation(fig, animate, range(frames.shape[0]))
    writergif = animation.PillowWriter(fps=15)
    # writermp4 = animation.FFMpegWriter(fps=15)
    ani.save("".join(path), writer=writergif)
    plt.close()
    # plt.show()


if __name__ == "__main__":
    for file in os.listdir("data/meta/"):
        meta = None
        with open(join("data/meta/", file),"r") as f:
            meta = json.load(f)
        visualize(meta["id"], meta)
        print("Visualized " + str(meta["id"]) + ".\n")


    # visualize("5c9954aa-3927-41f8-b698-2c1d06d31e45")