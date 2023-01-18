import numpy as np
import matplotlib.pyplot as plt


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
