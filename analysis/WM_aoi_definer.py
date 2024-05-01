#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:49:28 2024

@author: tomhawkins
"""

import matplotlib.pyplot as pyplot
from PIL import Image
import numpy
import glob
import cv2


from constants import *

# Points are identified as (TOP_LEFT), (TOP_RIGHT),
# (BOTTOM_LEFT), (BOTTOM_RIGHT)

aoi_points = {}

e_points = [(560, 350), (870, 350), (560,440), (870,440)]
n_points = [(670,440), (780, 440), (670,540), (780,540)]
m_points = [(630, 560), (815, 560), (630,640), (815, 640)]
ear_points = [(450,370), (500,370), (450, 440), (500, 440)]
f_points = [(540,250), (900, 250), (540,710), (900,710)]
disp_points = [(DISPSIZE[0], DISPSIZE[1])]

aoi_points['e_points'] = e_points
aoi_points['n_points'] = n_points
aoi_points['f_points'] = f_points
aoi_points['m_points'] = m_points
aoi_points['ear_points'] = ear_points


e_x_coordinates, e_y_coordinates = zip(*e_points)
n_x_coordinates, n_y_coordinates = zip(*n_points)
f_x_coordinates, f_y_coordinates = zip(*f_points)
m_x_coordinates, m_y_coordinates = zip(*m_points)
ear_x_coordinates, ear_y_coordinates = zip(*ear_points)
disp_x_coordinates, disp_y_coordinates = zip(*disp_points)


fig = pyplot.Figure()
ax = fig.add_axes([0, 0, 1, 1])

# Load the background image
fpath_img = os.path.join(IMGDIR, "CFD-WM.jpg")
img = cv2.imread(fpath_img)
scale_factor = 0.5
new_size = (round(img.shape[1]*scale_factor), round(img.shape[0]*scale_factor))
img_scaled = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

frame = numpy.zeros((DISPSIZE[1], DISPSIZE[0], 3), dtype=numpy.uint8)
y_si = (frame.shape[0] - img_scaled.shape[0])//2
y_ei = y_si + img_scaled.shape[0]
x_si = (frame.shape[1] - img_scaled.shape[1])//2
x_ei = x_si + img_scaled.shape[1]
frame[y_si:y_ei, x_si:x_ei, :] = img_scaled

frame = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
frame = frame.astype(numpy.float64) / 255.0

fig, ax = pyplot.subplots()


#Setting the limits of the plot to the screen size
ax.imshow(frame, \
extent=[x_si, x_ei, y_ei, y_si])
ax.set_ylim(DISPSIZE[1], 0)   
ax.set_xlim(0, DISPSIZE[0])

for key, points in aoi_points.items(): 
    pyplot.plot([points[0][0], points[1][0]], [points[0][1],points[0][1]],\
                 color='Green')
    pyplot.plot([points[1][0], points[1][0]], [points[0][1],points[2][1]],\
                 color='Green')
    pyplot.plot([points[0][0], points[1][0]], [points[2][1],points[2][1]],\
                 color='Green')
    pyplot.plot([points[0][0], points[0][0]], [points[0][1],points[2][1]],\
                 color='Green')

pyplot.show()
