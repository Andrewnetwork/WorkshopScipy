# MNIST_3D_VIS.py
# Andrew Ribeiro @ AndrewRib.com
# In order to understand how we can think of mnist data in 3d, we must
# understand how we have different 2d projections, hyperplanes, in that space.
# It helps a great deal if we can explore the data in 3d. This is a tool for this exploration.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as off
from plotly.graph_objs import Surface

import pandas as pd
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

unique, counts    = np.unique(mnist.train.labels, return_counts=True)
sortedCount       = sorted(dict(zip(unique, counts)).items(), key=lambda x: x[1],reverse=True)
sortedCountLabels = [i[0] for i in sortedCount]
sortedCountFreq   = [i[1] for i in sortedCount]

# TODO: Make more efficient.
# First we will zip the training labels with the training images
dataWithLabels = zip(mnist.train.labels, mnist.train.images)

# Now let's turn this into a dictionary where subsets of the images in respect
# to digit class are stored via the corresponding key.

# Init dataDict with keys [0,9] and empty lists.
digitDict = {}
for i in range(0,10):
    digitDict[i] = []

# Assign a list of image vectors to each corresponding digit class index.
for i in dataWithLabels:
    digitDict[i[0]].append(i[1])

# Convert the lists into numpy matricies. (could be done above, but I claim ignorace)
for i in range(0,10):
    digitDict[i] = np.matrix(digitDict[i])
    print("Digit {0} matrix shape: {1}".format(i,digitDict[i].shape))


nImgs = digitDict[9].shape[0]
avgImg = np.dot(digitDict[9].T, np.ones((nImgs,1)))/nImgs

data = [
    go.Surface(
        z= avgImg.reshape(28,28)
    )
]
layout = go.Layout(
    title='Zero',
    autosize=True,
    width=1000,
    height=1000,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)


plane_z_values = np.empty([28,28])
plane_z_values.fill(0.4)
surf = [ go.Surface(z=plane_z_values) ]


#fig = go.Figure(data=surf ,layout=layout)
fig = go.Figure(data=[dict(z=plane_z_values, type='surface'),dict(z=avgImg.reshape(28,28), type='surface')] ,layout=layout)
off.plot(fig, show_link=False)

#dict(z=surf, type='surface')
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot(avgImg.reshape(28,28))
#plt.show()

# downscaling has a "smoothing" effect
#lena = scipy.misc.imresize(lena, 0.15, interp='cubic')

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:28, 0:28]

# create the figure
# fig = plt.figure(figsize=(14,14))
# ax = fig.gca(projection='3d',axisbg='gray')
# ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
# ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
# ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
#
# plane_z_values = np.empty([28,28])
# plane_z_values.fill(0.4)
# print(plane_z_values.shape)
# ax.plot_surface(xx, yy, plane_z_values, cmap=plt.cm.Reds,
#         linewidth=10)
#
# ax.plot_surface(xx, yy, avgImg.reshape(28,28), cmap=plt.cm.Reds,
#         linewidth=0.2)
# plt.title("Digit 0")
#
#
# plt.show()

