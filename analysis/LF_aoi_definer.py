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

# Points are identified as 
# (TOP_LEFT), (TOP_RIGHT),
# (BOTTOM_LEFT), (BOTTOM_RIGHT)
aoi_points = {}


e_points = [(550, 370), (880, 370), (550,450), (880,450)]
n_points = [(660, 450), (790, 450), (660,540), (790,540)]
f_points = [(530, 250), (910, 250), (530,700), (910,700)]
m_points = [(620, 560), (830, 560), (620,630), (830, 630)]
disp_points = [(DISPSIZE[0], DISPSIZE[1])]


aoi_points['e_points'] = e_points
aoi_points['n_points'] = n_points
aoi_points['f_points'] = f_points
aoi_points['m_points'] = m_points


e_x_coordinates, e_y_coordinates = zip(*e_points)
n_x_coordinates, n_y_coordinates = zip(*n_points)
f_x_coordinates, f_y_coordinates = zip(*f_points)
m_x_coordinates, m_y_coordinates = zip(*m_points)
disp_x_coordinates, disp_y_coordinates = zip(*disp_points)


fig = pyplot.Figure()
ax = fig.add_axes([0, 0, 1, 1])

# Load the background image
fpath_img = os.path.join(IMGDIR, "CFD-LF.jpg")
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
