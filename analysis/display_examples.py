#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:17:06 2024

@author: tomhawkins
"""

from matplotlib import pyplot
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import numpy
from PIL import Image
import glob
import cv2

from constants import *

# Lists for display loops
pup_display = ['black', 'white']
att_display = ['white_black', 'black_white']

# Create an example display of all the points in the calibration task. 
fig, ax = pyplot.subplots()
ax.set_ylim(DISPSIZE[1], 0)   
ax.set_xlim(0, DISPSIZE[0])

# Set background colour and remove axis
ax.set_facecolor('gray')

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Plot points
for px,py in CALIB_DOTS_PX:
    ax.plot(px, py, "o", markersize=5, \
        color="black")

# Save and close plot
fig.savefig(os.path.join(CALDIR,"Calibration_example.png"))
pyplot.close()  

# Create example of the pup trials. 

#Fixation cross settings
cross_size = 10
center_x = DISPSIZE[0] / 2
center_y = DISPSIZE[1] / 2

for display in pup_display:
    fig, ax = pyplot.subplots()
    ax.set_ylim(DISPSIZE[1], 0)   
    ax.set_xlim(0, DISPSIZE[0])
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    ax.set_facecolor(display)
    
    ax.plot([center_x - cross_size, center_x + cross_size], \
        [center_y, center_y], color='grey', linewidth=2)
    ax.plot([center_x, center_x], [center_y - cross_size,\
        center_y + cross_size], color='grey', linewidth=2)
    fig.savefig(os.path.join(PUPDIR,"example_display_{}.png".format(display)))
    pyplot.close()  


circle_size = DISK_RADIUS*2
eccentricity = eccentricity

# Create the att_pupil screens 
for display in att_display: 
    fig, ax = pyplot.subplots()
    ax.set_ylim(DISPSIZE[1], 0)   
    ax.set_xlim(0, DISPSIZE[0])
    
    #set background colour and remove axis
    ax.set_facecolor('grey') 
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    #Plot the centre fixation
    ax.plot([center_x - cross_size, center_x + cross_size], \
        [center_y, center_y], color='black', linewidth=2)
    ax.plot([center_x, center_x], [center_y - cross_size, \
        center_y + cross_size], color='black', linewidth=2)
    
    # Set disk colours
    if display == 'white_black':
        color1 = 'white'
        color2 = 'black'
    else:
        color1 = 'black'
        color2 = 'white'
    
    # Draw the circles
    ellipse1 = Ellipse((center_x - eccentricity, center_y), \
        width=circle_size, height=circle_size,\
        edgecolor=color1, facecolor=color1)
    ellipse2 = Ellipse((center_x + eccentricity, center_y), \
        width=circle_size, height=circle_size,\
        edgecolor=color2, facecolor=color2)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    
    # Save pre target displays to ATT output directory
    fig.savefig(os.path.join(ATTDIR,\
        "example_display_pre_target{}.png".format(display)))
    
    # Draw the Target onto the screen
    t_width = 40
    t_height = 60
    
    
    if display == 'white_black':
        # horizontal line
        ax.plot([center_x - (eccentricity+t_width/2), center_x - (eccentricity\
            - t_width/2)], [center_y +t_height/2, center_y+t_height/2], \
            color='grey', linewidth=2)  
        # vertical line
        ax.plot([center_x - eccentricity, center_x - eccentricity], [center_y \
            - t_height/2, center_y + t_height/2], color='grey', linewidth=2)  
        # Save plot
        fig.savefig(os.path.join(ATTDIR,"example_display_{}_t_down.png"\
            .format(display)))
    
    else:
        # horizontal line
        ax.plot([center_x +(eccentricity+t_width/2), center_x + (eccentricity\
            - t_width/2)], [center_y -t_height/2, center_y-t_height/2], \
            color='grey', linewidth=2)  
        # vertical line
        ax.plot([center_x + eccentricity, center_x + eccentricity], [center_y \
            - t_height/2, center_y + t_height/2], color='grey', linewidth=2)  
        # Save plot
        fig.savefig(os.path.join(ATTDIR,"example_display_{}_t_up.png"\
            .format(display)))
    pyplot.close()

# Grey trial start screen
fig, ax = pyplot.subplots()
ax.set_ylim(DISPSIZE[1], 0)   
ax.set_xlim(0, DISPSIZE[0])

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

ax.set_facecolor('grey')

ax.plot([center_x - cross_size, center_x + cross_size], \
    [center_y, center_y], color='black', linewidth=2)
ax.plot([center_x, center_x], [center_y - cross_size,\
    center_y + cross_size], color='black', linewidth=2)
fig.savefig(os.path.join(ATTDIR,"example_display_grey.png"))
pyplot.close()  

    
# AOI image examples 

# Load images
img_files = {}
img_files['m_img'] = os.path.join(IMGDIR, "CFD-WM.jpg")

img_files['f_img'] = os.path.join(IMGDIR, "CFD-LF.jpg")

for img in img_files:
    
    # Points are identified as (TOP_LEFT), (TOP_RIGHT),
    # (BOTTOM_LEFT), (BOTTOM_RIGHT)
    aoi_points = {}
    
    if img == 'm_img':
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
        
    else:
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
    fpath_img = img_files[img]
    image = cv2.imread(fpath_img)
    scale_factor = 0.5
    new_size = (round(image.shape[1]*scale_factor), round(image.shape[0]*scale_factor))
    image_scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    frame = numpy.zeros((DISPSIZE[1], DISPSIZE[0], 3), dtype=numpy.uint8)
    y_si = (frame.shape[0] - image_scaled.shape[0])//2
    y_ei = y_si + image_scaled.shape[0]
    x_si = (frame.shape[1] - image_scaled.shape[1])//2
    x_ei = x_si + image_scaled.shape[1]
    frame[y_si:y_ei, x_si:x_ei, :] = image_scaled
    
    frame = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
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
    
    fig.savefig(os.path.join(AOIDIR,"AOI_example_{}.png".format(img)))

