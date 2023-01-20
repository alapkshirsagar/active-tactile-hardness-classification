import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib.animation as animation


def visualize(id):
    frames = np.load("data/frames/" + id + "_frames.npy")
    times = np.load("data/frames/" + id + "_times.npy")
    forces = np.load("data/forces/" + id + ".npy")
    path =  ["/home/ip-hard/Projects/ip-hardness/hardness-classification/processed_data/visualization/", str(id), ".mp4"]
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter("".join(path) , fourcc, 20, (320, 240), isColor = True)

    epsilon = 1400
    intensity = lambda x: np.linalg.norm(x)
    intensities = []
    for frame in frames:
        intensities.append(intensity(frame))

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)


    def animate(frame_num):
        timestamp = times[frame_num]

        ax0.cla()
        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax0.set_title("Frame " + str(frame_num))
        ax0.imshow(frames[frame_num, :, :, 0])

        ax1.set_title("Measured force")
        ax1.plot(forces[:, 0], forces[:, 1], color = "#d3d3d3")
        forcesSoFar = forces[:, 0] < timestamp
        ax1.plot(forces[forcesSoFar][:, 0], forces[forcesSoFar][:, 1], color = "blue")

        ax2.set_title("Frame intensity difference to 0th frame")
        ax2.plot(times, intensities - intensities[0], color = "#d3d3d3")
        ax2.plot(times[: frame_num], intensities[: frame_num] - intensities[0], color = "blue")
        ax2.hlines(epsilon, times[0], times[-1], colors = ["red"])

        ax3.imshow(frames[frame_num, :, :, 0])
    

    ani = animation.FuncAnimation(fig, animate, range(frames.shape[0]))
    writergif = animation.PillowWriter(fps=30)
    ani.save(id + str(".gif"), writer=writergif)

    plt.show()


if __name__ == "__main__":
    visualize("5c9954aa-3927-41f8-b698-2c1d06d31e45")

    
""" 

id = "5c9954aa-3927-41f8-b698-2c1d06d31e45"
idx = 30


frames = np.load("data/frames/" + id + "_frames.npy")
times = np.load("data/frames/" + id + "_times.npy")
forces = np.load("data/forces/" + id + ".npy")

frame_num = frames.shape[0]

neutral = frames[int(frame_num * .05)] # neural deformation before pressing

intensity = lambda x: np.mean(x)
dis = lambda x: np.linalg.norm( x- neutral)
distances = np.array(list(map(dis,frames)))
intensities = np.array(list(map(intensity, frames)))
min_intensity_index = np.argmin(intensities)
max_deformation_index = np.argmax(distances)
# print(max_deformation_index)

max_force_timestamp = forces[np.argmin(forces[:, 1]), 0]
max_force_index = np.argmin(np.absolute(times - max_force_timestamp))
print(min_intensity_index)

# plt.imshow(frames[min_intensity_index, :, :, 0] - frames[0, :, :, 0])
plt.plot(intensities)
plt.show()
 """