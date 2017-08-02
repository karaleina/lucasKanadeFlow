import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from sklearn import decomposition
import h5py
import math

filename = "4_oddech_przykrywka.avi"
filepath = os.path.abspath("C:\\Users\\ImioUser\\Desktop\\K&A\\pylon_spekle\\" + filename)
filename_save = "test.h5"

cap = cv2.VideoCapture(filepath)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,#100
                       qualityLevel = 0.005,#0.3
                       minDistance = 20,#7
                       blockSize = 7)#7

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

print("Znaleziona liczba śledzonych punktów"
      " = " + str(len(p0)))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Zmienne
all = []
mean_x_all_frames = []
mean_y_all_frames = []
angles_in_all_frames = []
speed_in_all_frames = []

# while(cap.isOpened()):
#     ret,frame = cap.read()
#
#     if(frame == None):
#         break
#
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#
#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]
#
#     # Here is the list of current changes in a and b for all piramids
#     current_a = []
#     current_b = []
#     current_c = []
#     current_d = []
#
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#
#         # Adding parameters to the list of current changes in a and b for all piramids
#         current_a.append(a)
#         current_b.append(b)
#         current_c.append(c)
#         current_d.append(d)
#
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#
#     # Adding current mean to the list of all means
#     mean_x_all_frames.append(np.mean(current_a))
#     mean_y_all_frames.append(np.mean(current_b))
#     angles_in_all_frames.append(math.atan((b-d)/float(a-c)))
#     speed_in_all_frames.append(math.sqrt(math.pow((b-d), 2) + math.pow((a-c),2)))
#
#     img = cv2.add(frame,mask)
#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
#
# cv2.destroyAllWindows()
# cap.release()
#
# ##########################################################
#
#

# # Write
# all = np.zeros(shape=(4,len(angles_in_all_frames)))
# all[2,:] = angles_in_all_frames
# all[3,:] = speed_in_all_frames
# all[0,:] = mean_x_all_frames
# all[1,:] = mean_y_all_frames
#
#
# with h5py.File(filename_save, 'w') as hf:
#     hf.create_dataset("all", data=all)

# Read
with h5py.File(filename_save, 'r') as hf:
    all = hf['all'][:]

print(all)
print(type(all))

mean_x = all[0,:]
mean_y = all[1,:]
angles_in_all_frames = all[2,:]
speed_in_all_frames = all[3,:]


# convert to matrix
train_data = np.mat([mean_x,mean_y])
pca_components = 1

# reduce both train and test data
pca = decomposition.PCA(n_components=pca_components).fit(train_data)
X_out_pca = pca.transform(train_data)

plt.subplot(2,2,1)
plt.plot(mean_x, mean_y)
plt.title("Średni ruch spekli w wejściowych współrzędnych")

plt.subplot(2,2,2)
plt.plot(X_out_pca)
plt.title("Ruch spekli po PCA")

plt.subplot(2,2,4)
plt.plot(angles_in_all_frames)
plt.title("Zmiany kąta w czasie")

plt.subplot(2,2,3)
plt.plot(speed_in_all_frames)
plt.title("Zmiany prędkości w czasie")


plt.show()



