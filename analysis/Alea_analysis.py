#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:07:04 2024

@author: tomhawkins
"""

# Analysis code for the Alea eye-tracker comparison study. The experiment
# included a set of 5 benchmark tests (2 calibration tasks (start and end of
# experiment), free-viewing task, standard PLR pupillometry task and a covert
# attention pupillometry task). In addition to the Alea tracker, data was also 
# recorded for the EyeLink 1000 on the same tasks as a comparison. 

import os

from matplotlib import pyplot
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy
import scipy.stats
from scipy.ndimage import gaussian_filter
import glob
import pandas
import cv2
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.stats.anova import AnovaRM
import pingouin

from constants import *

from pygazeanalyser.aleareader import read_alea
from pygazeanalyser.edfreader import read_edf


# Downsample the EyeLink data?
DOWNSAMPLE_EYELINK = False

# Include the ear AOI?
INCLUDE_EAR = True

# SETTINGS
# Names of the eye trackers used to collect data.
trackers = ["Alea", "EyeLink"]

experiments = ['calibration', 'aoi', 'pupil', 'att_pupil']

# Number of calibration points.
n_calib_points = 12
# Calibration phases in the experiment.
calib_phases = ["calibration", "calibration_post"]
calib_measures_analyse = ["acc_ang", "rms_ang", "std_ang"]

# AOI images included
image_names = ["CFD-LF", "CFD-WM"]
# AOI Features
if INCLUDE_EAR == False:
    aoi_features = ["eyes", "nose", "mouth", "face", "other"]
else:
    aoi_features = ["eyes", "nose", "mouth", "face", "ear", "other"]

# Coordinates for aoi boxes for each image
if INCLUDE_EAR == False:
    aoi_boxes = {}
    aoi_boxes["CFD-WM"] = { \
        "eyes":{"x":560, "y":350, "w":310, "h":90}, \
        "nose":{"x":670, "y":440, "w":110, "h":100},\
        "mouth":{"x":630, "y":560, "w":185, "h":80},\
        "face":{"x":540, "y":250, "w":360, "h":460},\
        "other":{"x":0, "y":0, "w":DISPSIZE[0], "h":DISPSIZE[1]}}
    aoi_boxes["CFD-LF"] = { \
        "eyes":{"x":550, "y":370, "w":330, "h":80}, \
        "nose":{"x":660, "y":450, "w":130, "h":90},\
        "mouth":{"x":620, "y":560, "w":210, "h":70},\
        "face":{"x":530, "y":250, "w":380, "h":450},\
        "other":{"x":0, "y":0, "w":DISPSIZE[0], "h":DISPSIZE[1]}}
else:
    aoi_boxes = {}
    aoi_boxes["CFD-WM"] = { \
        "eyes":{"x":560, "y":350, "w":310, "h":90}, \
        "nose":{"x":670, "y":440, "w":110, "h":100},\
        "mouth":{"x":630, "y":560, "w":185, "h":80},\
        "face":{"x":540, "y":250, "w":360, "h":460},\
        "ear":{"x":450, "y":370, "w":50, "h":70},\
        "other":{"x":0, "y":0, "w":DISPSIZE[0], "h":DISPSIZE[1]}}
    aoi_boxes["CFD-LF"] = { \
        "eyes":{"x":550, "y":370, "w":330, "h":80}, \
        "nose":{"x":660, "y":450, "w":130, "h":90},\
        "mouth":{"x":620, "y":560, "w":210, "h":70},\
        "face":{"x":530, "y":250, "w":380, "h":450},\
        "ear":{"x":0, "y":900, "w":0, "h":0},\
        "other":{"x":0, "y":0, "w":DISPSIZE[0], "h":DISPSIZE[1]}}

# PUP stimuli
pupil_stimuli = ["dark", "light"]
# number of pupil trials 
n_trials = 50

# att_pup conditions
attend_cond = ["attend_light", "attend_dark"]
target_cong = ["valid", "invalid"]

# Start and stop message for each experiment.
start_msg = { \
    "calibration": "CALIBRATION_TRIALSTART", \
    "AOI": "IMAGE_TRIALSTART", \
    "pupil": "PUPIL_TRIALSTART", \
    "att_pupil": "ATT_PUPIL_TRIALSTART"}
stop_msg = { \
    "calibration": "CALIB_TRIALSTOP", \
    "AOI": "IMAGE_TRIALSTOP", \
    "pupil": "PUPIL_ITISTOP", \
    "att_pupil": "ATT_PUPIL_TRIALSTOP"}

# Create a few colour for plotting.
plotcols = {}
for i in range(n_calib_points):
    plotcols["target_nr_{}".format(i)] = numpy.random.rand(3)

for img_name in image_names:
    plotcols["{}".format(img_name)] = numpy.random.rand(3)

# Find all data files.
data_files = {}
# Use glob to get a list of TSV files in the folder. These are Alea files.
data_files["Alea"] = glob.glob(DATADIR + '/*.tsv')
data_files["Alea"].sort()
# Find EyeLink files.
data_files["EyeLink"] = glob.glob(DATADIR + '/*.asc')
data_files["EyeLink"].sort()

# Run through file paths to find participant codes.
ppnames = []
for tracker in trackers:
    for fname in data_files[tracker]:
        name, ext = os.path.splitext(os.path.basename(fname))
        p, nr, tracker = name.split("_")
        ppname = "{}_{}".format(p, nr)
        if ppname not in ppnames:
            ppnames.append(ppname)
ppnames.sort()

# Count the number of participants and trackers.
n_participants = len(ppnames)
n_trackers = len(trackers)
n_calib_phases = len(calib_phases)
n_calib_samples = CALIB_DOT_DUR
n_images = len(image_names)
n_aoi_features = len(aoi_features)
n_image_samples = IMGTIME
n_image_samples_alea = int(IMGTIME/ALEA_SAMPLE_MS)
n_pupil_stimuli = len(pupil_stimuli)
n_pupil_samples = int(PUPTRIALTIME / EYELINK_SAMPLE_MS)
n_attend_cond = len(attend_cond)
n_target_cong = len(target_cong)
n_pupil_samples_alea = int(PUPTRIALTIME / ALEA_SAMPLE_MS)
n_calibration_samples_alea = int(CALIB_DOT_DUR/ALEA_SAMPLE_MS)

# Create an empty dict for each experiment.
results = {}

# Create empty dicts for the first calibration.
shape = (n_participants, n_trackers, n_calib_phases, n_calib_points)
results["calibration"] = {}
for var in ["acc", "rms", "std", "acc_ang", "rms_ang", "std_ang"]:
    results["calibration"][var] = numpy.zeros(shape, dtype=numpy.float64) \
        * numpy.nan

# Create additional results frames to create the averaged plots 
shape = (n_participants, n_trackers, n_calib_phases, n_calib_points, n_calib_samples)
results["calibration"]["x"] = {}
results["calibration"]["y"] = {}

results["calibration"]["x"] = numpy.zeros(shape, dtype=numpy.float64) \
    * numpy.nan
results["calibration"]["y"] = numpy.zeros(shape, dtype=numpy.float64) \
    * numpy.nan


# Create empty dicts for the AOI.
shape = (n_participants, n_trackers, n_images)
results["aoi"] = {}
results["aoi"]["fix_nr"] = {}
results["aoi"]["dwell"] = {}
results["aoi"]["proportion"] = {}

if INCLUDE_EAR == False:
    for var in ["eyes", "nose", "mouth", "face", "other"]:
        results["aoi"]["fix_nr"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan
    for var in ["eyes", "nose", "mouth", "face", "other"]:
        results["aoi"]["dwell"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan
    for var in ["eyes", "nose", "mouth", "face", "other"]:
        results["aoi"]["proportion"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan
else:
    for var in ["eyes", "nose", "mouth", "face", "ear", "other"]:
        results["aoi"]["fix_nr"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan
    for var in ["eyes", "nose", "mouth", "face", "ear", "other"]:
        results["aoi"]["dwell"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan
    for var in ["eyes", "nose", "mouth", "face", "ear", "other"]:
        results["aoi"]["proportion"][var] = numpy.zeros(shape, dtype=int) \
            * numpy.nan

# Create empty dicts for aoi x, y coordinates for averaged heat map plot 
shape = (n_participants, n_trackers, n_images, n_image_samples)
results["aoi"]["x"] = {}
results["aoi"]["y"] = {}

results["aoi"]["x"] = numpy.zeros(shape, dtype=numpy.float64) \
    * numpy.nan
results["aoi"]["y"] = numpy.zeros(shape, dtype=numpy.float64) \
    * numpy.nan

# Empty dict for Pupil and ATT_pupil
shape = \
    (n_participants, n_trackers, n_pupil_stimuli, n_trials, n_pupil_samples)
results["pupil"] = {}
for var in ["raw", "corrected", "corrected_baseline-divided"]:
    results["pupil"][var] = numpy.zeros(shape, dtype=numpy.float64) \
        * numpy.nan

shape = \
    (n_participants, n_trackers, n_attend_cond, n_trials, n_pupil_samples)
results["att_pupil"] = {}
for var in ["raw", "corrected", "corrected_baseline-divided"]:
    results["att_pupil"][var] = numpy.zeros(shape, dtype=numpy.float64) \
        * numpy.nan

shape = \
    (n_participants, n_trackers, n_attend_cond, n_trials, n_target_cong)
for var in ["reaction_time"]:
    results["att_pupil"][var] = numpy.zeros(shape, dtype=numpy.float64) \
        * numpy.nan

# Run through eye trackers.
for ti, tracker in enumerate(trackers):
    
    print("Processing data for {} tracker".format(tracker))
    
    # Loop through all data files for this tracker.
    for fi, fpath in enumerate(data_files[tracker]):
        
        # Parse the file name.
        name, ext = os.path.splitext(os.path.basename(fpath))
        p, nr, _ = name.split("_")
        ppname = "{}_{}".format(p, nr)
        # Get the index number for this participant.
        ppi = ppnames.index(ppname)

        print("Loading data for participant {} ({}/{})".format(ppname, ppi+1, \
            n_participants))
        
        # Loop through all experiments.
        for exp_name in start_msg.keys():
            
            # Load the data.
            if tracker == "Alea":
                data = read_alea(fpath, start=start_msg[exp_name], \
                    stop=stop_msg[exp_name])
            elif tracker == "EyeLink":
                data = read_edf(fpath, start=start_msg[exp_name], \
                    stop=stop_msg[exp_name])

            # Clean the data.
            for i in range(len(data)):
                # Convenience renaming.
                x = data[i]["x"]
                y = data[i]["y"]
                pup = data[i]["size"]
    
                # Remove missing data.
                x[x==0] = numpy.nan
                y[y==0] = numpy.nan
                pup[pup==0] = numpy.nan
                
                # Remove blinks and saccades if we have the data.
                if tracker == "EyeLink":
                    for event in data[i]["events"]["Eblk"] \
                        + data[i]["events"]["Esac"]:
                        si = numpy.argmin(numpy.abs( \
                            data[i]["trackertime"] - event[0]))
                        ei = numpy.argmin(numpy.abs( \
                            data[i]["trackertime"] - event[1]))
                        x[si:ei] = numpy.nan
                        y[si:ei] = numpy.nan
                        pup[si:ei] = numpy.nan
                
                # TODO: Cut out blinks accurately
                # There are undetected blinks in the EyeLink data, so we have to
                # manually detect and exclude them. We also include a 50 ms
                # buffer on either side, because the rapid decrease and increase
                # in estimated pupil size before and after a blink are what is
                # causing the artefacts.
                if tracker == "EyeLink":
                    blink = (pup == 0) | numpy.isnan(pup)
                    transitions = numpy.diff(blink.astype(numpy.int8))
                    onsets = numpy.where(transitions == 1)[0]
                    offsets = numpy.where(transitions == -1)[0]
                    for bi in range(onsets.shape[0]):
                        si = max(0, onsets[bi]-50)
                        if bi >= offsets.shape[0]:
                            ei = pup.shape[0]
                        else:
                            ei = min(pup.shape[0], offsets[bi]+50)
                        x[si:ei] = numpy.nan
                        y[si:ei] = numpy.nan
                        pup[si:ei] = numpy.nan
            
                # Downsample the EyeLink data.
                if tracker == "EyeLink" and DOWNSAMPLE_EYELINK:
                    indices = numpy.arange(0, x.shape[0], 25)
                    x = x[indices]
                    y = y[indices]
                    pup = pup[indices]
                    data[i]["time"] = data[i]["time"][indices]
                    data[i]["trackertime"] = data[i]["trackertime"][indices]
        
            # CALIBRATION EXPERIMENTS
            if exp_name == "calibration":
    
                # Create a new figure to plot this participant's samples in.
                fig = {}
                ax = {}
                for calibration_phase in calib_phases:
                    fig[calibration_phase] = pyplot.Figure()
                    ax[calibration_phase] = \
                        fig[calibration_phase].add_axes([0, 0, 1, 1])
                
                
                # Go through all trials.
                for i in range(len(data)):
                    
                    # Extract information from the logged messages, including trial
                    # settings and timing.
                    calibration_phase = None
                    target_nr = None
                    target_x = None
                    target_y = None
                    t0 = None
                    t_onset = None
                    t_fixation = None
                    t_offset = None
                    for (t, msg) in data[i]["events"]["msg"]:
                        if start_msg["calibration"] in msg:
                            # The trial message is formatted like this:
                            # CALIBRATION_TRIALSTART, phase=calibration_post, trialnr=2, x=144, y=270
                            # We thus split by commas to extract the trial info.
                            trial_msg = msg
                            _, calibration_phase, target_nr, target_x, target_y = \
                                trial_msg.split(",")
                            calibration_phase = \
                                calibration_phase[calibration_phase.find("=")+1:]
                            i_phase = calib_phases.index(calibration_phase)
                            target_nr = int(target_nr[target_nr.find("=")+1:])
                            target_x = float(target_x[target_x.find("=")+1:])
                            target_y = float(target_y[target_y.find("=")+1:])
                        elif "CALIB_BLANK_SCREEN" in msg:
                            t0 = t
                        elif "CALIB_DOT_ONSET" in msg:
                            t_onset = t
                        elif "CALIB_FIXATION_DETECTED" in msg:
                            t_fixation = t
                        elif "CALIB_TRIALSTOP" in msg:
                            t_offset = t
                    
                    # The end message won't be included (this is due to how the
                    # reader function works), so set the offset to the very last
                    # sample.
                    if t_offset is None:
                        t_offset = data[i]["trackertime"][-1]
                    
                    # Find the data from around the target fixation.
                    i_onset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - t_onset))
                    i_fixation = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - t_fixation))
                    i_offset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - t_offset))
                    
                    # Convenience renaming (for less typing later).
                    x = data[i]["x"][i_fixation:i_offset]
                    y = data[i]["y"][i_fixation:i_offset]
                    pup = data[i]["size"][i_fixation:i_offset]
                    
                    # Calculate distance from target for each sample
                    d_target = numpy.sqrt((x - target_x)**2 + (y - target_y)**2)
                    
                    # Set samples that are too far from the target to NaN
                    x[d_target > 72] = numpy.nan
                    y[d_target > 72] = numpy.nan
                    
                    
                    if x.shape[0] == n_calib_samples:
                        results["calibration"]["x"][ppi, ti, i_phase, target_nr, :] = x
                        results["calibration"]["y"][ppi, ti, i_phase, target_nr, :] = y
                        
                    elif x.shape[0] < n_calib_samples:
                        ei = x.shape[0]
                        results["calibration"]["x"][ppi, ti, i_phase, target_nr, :ei] = \
                        x
                        results["calibration"]["y"][ppi, ti, i_phase, target_nr, :ei] = \
                        y
                    else:
                        results["calibration"]["x"][ppi, ti, i_phase, target_nr, :] = \
                            x[:n_calib_samples]
                        results["calibration"]["y"][ppi, ti, i_phase, target_nr, :] = \
                            y[:n_calib_samples]

                    # Compute the distance between target and samples.
                    d_target = numpy.sqrt((x-target_x)**2 + (y-target_y)**2)
                    # Kick out samples that are obviously too far? (I.e. someone
                    # made a saccade elsewhere.)
                    d_target[d_target>72] = numpy.nan
                    
                    # Convert pixel coordinates into degrees of visual angle
                    d_target_ang = d_target * DEG_PER_PX

                    
                    # Compute accuracy as the average absolute distance.
                    acc = numpy.nanmean(numpy.abs(d_target))
                    # convert to angle 
                    acc_ang = acc * DEG_PER_PX
                    # Compute RMS noise as the average distance between consecutive
                    # samples.
                    d_intersample = numpy.sqrt(numpy.diff(x)**2 + numpy.diff(y)**2)
                    rms = numpy.nanmean(d_intersample)
                    # convert to angle 
                    rms_ang = rms * DEG_PER_PX
                    
                    # Compute the standard deviation as the dispersion around a 
                    # centriod for a sequence of gaze positions. 
                    # Calculate squared difference for all x and y points
                    std = numpy.sqrt(numpy.nanstd(x)**2 + numpy.nanstd(y)**2)
                    # convert to angle 
                    std_ang = std * DEG_PER_PX
                    
                    # Save the accuracy and RMS noise.
                    results["calibration"]["acc"][ppi, ti, i_phase, target_nr] = acc
                    results["calibration"]["rms"][ppi, ti, i_phase, target_nr] = rms
                    results["calibration"]["std"][ppi, ti, i_phase, target_nr] = std
                    
                    
                    results["calibration"]["acc_ang"][ppi, ti, i_phase, target_nr] = acc_ang
                    results["calibration"]["rms_ang"][ppi, ti, i_phase, target_nr] = rms_ang
                    results["calibration"]["std_ang"][ppi, ti, i_phase, target_nr] = std_ang

                    # Plot this participant's samples.
                    ax[calibration_phase].plot(x, y, "o", \
                        color=plotcols["target_nr_{}".format(target_nr)], alpha=0.1)
                    # Plot the associated target.
                    ax[calibration_phase].plot(target_x, target_y, "o", markersize=10, \
                        color=plotcols["target_nr_{}".format(target_nr)], alpha=0.5)
                
                # Finish the calibration plot.
                for calibration_phase in calib_phases:
                    ax[calibration_phase].set_xlim(0, DISPSIZE[0])
                    ax[calibration_phase].set_ylim(0, DISPSIZE[1])
                    fig[calibration_phase].\
                        savefig(os.path.join(CALDIR,"calibration_{}_{}_{}.png"\
                        .format(ppname, tracker, calibration_phase)))
                    pyplot.close(fig[calibration_phase])
             

            # FREEVIEWING EXPERIMENT
            elif exp_name == "AOI":
    
                # Initialize dictionaries/lists to store hits for 
                # each AOI image separately
                if INCLUDE_EAR == False:
                    hits_per_aoi = {aoi_image_name: {"eyes": [], "nose": [],\
                        "mouth": [], "face": [], "other": []}\
                        for aoi_image_name in image_names}
                else:
                    hits_per_aoi = {aoi_image_name: {"eyes": [], "nose": [],\
                        "mouth": [], "face": [], "ear": [], "other": []}\
                        for aoi_image_name in image_names}

                # Run through all trials.
                for i in range(len(data)):
                    
                    # Extract information from the logged messages, including 
                    # trial settings and timing.
                    aoi_image_name = None
                    t0 = None
                    img_onset = None
                    img_offset = None
                    for (t, msg) in data[i]["events"]["msg"]:
                        if start_msg["AOI"] in msg:
                            # The trial message is formatted like this:
                            # IMAGE_TRIALSTART, imgname=CFD-WM.jpg
                            # We thus split by commas to extract the trial info.
                            trial_msg = msg
                            _, pic_name = trial_msg.split(",")
                            start_index = pic_name.find("=")+1
                            end_index = pic_name.find(".jpg")
                            aoi_image_name = pic_name[start_index:end_index]
                            img_index = image_names.index(aoi_image_name)
                        elif "IMGE_FIXATION" in msg:
                            t0 = t
                        elif "IMAGE_ONSET" in msg:
                            img_onset = t
                        elif "IMGAE_OFFSET" in msg:
                            img_offset = t
                    # The end message won't be included (this is due to how the
                    # reader function works), so we set the offset to the very last
                    # sample.
                    if img_offset is None:
                        img_offset = data[i]["trackertime"][-1]
                    
                    # Find the data from the stimuli presentation.
                    i_img_onset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - img_onset))
                    i_img_offset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - img_offset))
                    
                    # Convenience renaming (for less typing later).
                    x = data[i]["x"][i_img_onset:i_img_offset]
                    y = data[i]["y"][i_img_onset:i_img_offset]
                    pup = data[i]["size"][i_img_onset:i_img_offset]

                    if x.shape[0] == n_image_samples:
                        results["aoi"]["x"][ppi, ti, img_index, :] = x
                        results["aoi"]["y"][ppi, ti, img_index, :] = y
                        
                    elif x.shape[0] < n_image_samples:
                        ei = x.shape[0]
                        results["aoi"]["x"][ppi, ti, img_index, :ei] = \
                        x
                        results["aoi"]["y"][ppi, ti, img_index, :ei] = \
                        y
                    else:
                        results["aoi"]["x"][ppi, ti, img_index, :] = \
                            x[:n_image_samples]
                        results["aoi"]["y"][ppi, ti, img_index, :] = \
                            y[:n_image_samples]
                     
                    #Define hits and dwell dicts 
                    hits = {}
                    n_hits = {}
                    dwell = {}
                    has_not_been_counted = numpy.ones(x.shape, dtype=bool)
                    
                    for aoi_name in aoi_features:
                        rect = aoi_boxes[aoi_image_name][aoi_name]
                        hor = (x > rect["x"]) & (x < rect["x"]+rect["w"])
                        ver = (y > rect["y"]) & (y < rect["y"]+rect["h"])
                        hits[aoi_name] = hor & ver & has_not_been_counted
                        has_not_been_counted[hits[aoi_name]] = False
                        
                        n_hits[aoi_name] = numpy.sum(hits[aoi_name])
                        if tracker == "Alea":
                            sample_dur = ALEA_SAMPLE_MS
                        elif tracker == "EyeLink":
                            sample_dur = EYELINK_SAMPLE_MS
                        dwell[aoi_name] = n_hits[aoi_name] * sample_dur
                         
                        # Save the data in our dict/array holding variables.
                        results["aoi"]["fix_nr"][aoi_name][ppi, ti, img_index] \
                            = n_hits[aoi_name]
                        results["aoi"]["dwell"][aoi_name][ppi, ti, img_index] \
                            = dwell[aoi_name]
                        
                    total_hits = sum(n_hits.values()) 
                    
                    proportions = {key: value / total_hits \
                                   for key, value in n_hits.items()}
                    
                    for feature in proportions:
                        results["aoi"]["proportion"][feature][ppi, ti, img_index]\
                            = proportions[feature]
                        
            elif exp_name == "pupil":
                
                baseline = None
                
                for i in range(0, len(data)):
                
                    # Extract information from the logged messages, including 
                    # trial settings and timing.
                    trial_stim = None
                    t0 = None
                    baseline_start = None
                    baseline_end = None
                    trial_onset = None
                    trial_offset = None
                    t_n = None
                
                    baseline_duration_ms = 500  
                    if tracker == "Alea":
                        baseline_duration_samples = \
                            int(baseline_duration_ms / ALEA_SAMPLE_MS)
                    else:
                        baseline_duration_samples = \
                            int(baseline_duration_ms / EYELINK_SAMPLE_MS)

                    for (t, msg) in data[i]["events"]["msg"]:
                        if start_msg["pupil"] in msg:
                            # The trial message is formatted like this:
                            # PUPIL_TRIALSTART, stimulus=dark
                            # We thus split by commas to extract the trial info.
                            trial_msg = msg
                            _, stimulus = trial_msg.split(",")
                            trial_stim = \
                                stimulus[stimulus.find("=")+1:]
                            trial_stim = trial_stim.replace("\n", "")
                            si = ["dark", "light"].index(trial_stim)
                            t0 = t
                            
                            trial_onset = t
                            baseline_end = t
                        elif "PUPIL_TRIALSTOP" in msg:
                            trial_offset = t
                        
                    # Trial number
                    t_n = i
                    
                    # Skip the trial if no messages were recorded.
                    if (trial_onset is None) or (trial_offset is None):
                        print(("\tWARNING: Not all messages recorded for " \
                           + "{}, trial {}!").format(exp_name, i))
                        continue
                    
                    
                    # Find the trial data onset/offset indices.
                    i_trial_onset = numpy.argmin(numpy.abs(\
                        data[i]["trackertime"] - trial_onset))
                    i_trial_offset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - trial_offset))
                        
                    # Compute and store the normalised pupil data.
                    if baseline is not None:
                        # Convenience renaming of pupil data.
                        pup = data[i]["size"][i_trial_onset:i_trial_offset]
                        # Check which has the fewest samples: our target
                        # array, or the current pupil size array.
                        if pup.shape[0] == n_pupil_samples:
                            results["pupil"]["raw"][ppi, ti, si, t_n, :] = pup
                        elif pup.shape[0] < n_pupil_samples:
                            ei = pup.shape[0]
                            results["pupil"]["raw"][ppi, ti, si, t_n, :ei] = \
                            pup
                        else:
                            results["pupil"]["raw"][ppi, ti, si, t_n, :] = \
                                pup[:n_pupil_samples]
                        # Compute the average pupil size in the baseline 
                        # period.
                        baseline_avg = numpy.nanmean(baseline)
                        # If all baseline samples were NaN, the nanmean is also
                        # a NaN (mean of empty slice). This means trial data 
                        # can't be corrected.
                        if not numpy.isnan(baseline_avg):
                            results["pupil"]["corrected"][ppi, ti, si, t_n, :]\
                                = results["pupil"]["raw"][ppi, ti, si, t_n, :]\
                                - baseline_avg
                            results["pupil"]["corrected_baseline-divided"]\
                                [ppi, ti, si, t_n, :] = \
                                results["pupil"]["raw"][ppi, ti, si, t_n, :] \
                                / baseline_avg
                        else:
                            print("\tOOPS, all NaNs! Trial {} for {}".format( \
                                i, exp_name))
                    
                    # Define the baseline start for the next trial as several 
                    # samples into the current trial.
                    i_baseline_start = i_trial_offset \
                        - baseline_duration_samples

                    # Save the baseline for the next trial.
                    baseline = pup[i_baseline_start:i_trial_offset]
                    
            elif exp_name == "att_pupil":
                
                # Preprepare dicts to hold figures and axes in.
                fig = {}
                ax = {}
               
                for condition in attend_cond:
                    fig[condition] = pyplot.Figure()
                    ax[condition] = \
                        fig[condition].add_axes([0, 0, 1, 1])
                
                for i in range(len(data)):
                
                    # Extract information from the logged messages, including 
                    # trial settings and timing.
                    disk_position = None
                    attend_left = None
                    target_left = None
                    target_up = None
                    target_valid = None
                    t0 = None
                    baseline_start = None
                    baseline_end = None
                    trial_onset = None
                    trial_offset = None
                    t_n = None
                    attend_stim = None
                    target_correct = None
                    response_start = None
                    response_end = None
                    
                    for (t, msg) in data[i]["events"]["msg"]:
                        if start_msg["att_pupil"] in msg:
                            # The trial message is formatted like this:
                            # ATT_PUPIL_TRIALSTART, disk_position=dark_right,
                            # attend_left=0, target_left=0, 
                            # target_up=1, target_valid=1
                            # We thus split by commas to extract the trial info.
                            trial_msg = msg
                            _, disk_position, attend_left, target_left,\
                                target_up, target_valid = trial_msg.split(",")
                            disk_position = \
                                disk_position[disk_position.find("=")+1:]
                            attend_left = \
                                attend_left[attend_left.find("=")+1:]
                            target_left = \
                                target_left[target_left.find("=")+1:]
                            target_up = \
                                target_up[target_up.find("=")+1:]
                            target_valid = \
                                target_valid[target_valid.find("=")+1:]
                            target_valid = target_valid.replace("\n", "")
                            
                            baseline_start = t
                            t0 = t
                            
                        elif "ATT_PUPIL_CUE_SOUND_ONSET" in msg:
                            baseline_end = t
                            trial_onset = t
                        elif "ATT_PUPIL_TARGET_ONSET" in msg:
                            response_start = t
                        elif "ATT_PUPIL_TRIALSTOP" in msg:
                            # Message :
                            # ATT_PUPIL_TRIALSTOP, response=up, correct=1
                            trial_msg = msg
                            _, response, target_correct = trial_msg.split(",")
                            target_correct = \
                                target_correct[target_correct.find("=")+1:]
                            
                            trial_offset = t
                            response_end = t
                            
                    # trial number
                    t_n = i
                    
                    if baseline_start == None:
                        baseline_start = data[i]["trackertime"][0]
                        
                    if trial_offset == None:
                        trial_offset = data[i]["trackertime"][-1]
                    if response_end == None:
                        response_end = data[i]["trackertime"][-1]
                    
                    # Skip the trial if no messages were recorded.
                    if (baseline_start is None) or (trial_offset is None):
                        print(("\tWARNING: Not all messages recorded for " \
                           + "{}, trial {}!").format(exp_name, i))
                        continue
                    
                    # Identify whether the trial had you look at a dark or
                    # light disk
                    if attend_left == '1' and disk_position == 'dark_left' or \
                        attend_left == '0' and disk_position == 'dark_right':
                        attend_stim = 'dark'
                    else:
                        attend_stim = 'light'
                        
                    # Identify whether the target is valid
                    if target_valid == 1: 
                        target_stim = 'valid'
                    else:
                        target_stim = 'invalid'
                    
                    # Identify if the response is correct
                    if target_correct == 1:
                        response = "correct"
                    else: 
                        response = "incorrect"
                    
                    #calculate reaction time 
                    reaction_time = response_end - response_start
                    
                    # Index the 2 attend conditions and whether the target was 
                    # valid
                    t_s = ["dark", "light"].index(attend_stim)
                    
                    t_t = ["valid", "invalid"].index(target_stim)
                    
                    
                    # find onset and offset indecies 
                    i_baseline_start = numpy.argmin(numpy.abs(\
                        data[i]["trackertime"] - baseline_start))
                    i_baseline_end = numpy.argmin(numpy.abs(\
                        data[i]["trackertime"] - baseline_end))
                    i_trial_onset = numpy.argmin(numpy.abs(\
                        data[i]["trackertime"] - trial_onset))
                    i_pupil_cutoff = numpy.argmin(numpy.abs(\
                        data[i]["trackertime"] - response_start))
                    i_trial_offset = numpy.argmin(numpy.abs( \
                        data[i]["trackertime"] - trial_offset))
                    
                    # Convenience renaming of x, y and pupil data.
                    base_pup = data[i]["size"][i_baseline_start:i_baseline_end]
                    trial_pup = data[i]["size"][i_trial_onset:i_pupil_cutoff]
                    x = data[i]["x"][i_trial_onset:i_pupil_cutoff]
                    y = data[i]["y"][i_trial_onset:i_pupil_cutoff]
                    
                    # Compute distance from fixation point.
                    x_fix = DISPSIZE[0]/2
                    y_fix = DISPSIZE[1]/2
                    gaze_fix_dist_px = numpy.sqrt((x-x_fix)**2 + (y-y_fix)**2)
                    # Convert distance from pixels to degrees of visual angle.
                    gaze_fix_dist_deg = px2deg(gaze_fix_dist_px, SCREENDIST, PX_PER_CM)
                    # NaN out all the samples that were too far from fixation.
                    too_far = gaze_fix_dist_deg > 2.5
                    trial_pup[too_far] = numpy.nan
                                        
                    # Check which has the fewest samples: target
                    # array, or the current pupil size array.
                    if trial_pup.shape[0] == n_pupil_samples:
                        results["att_pupil"]["raw"][ppi, ti, t_s, t_n, :]\
                            = trial_pup
                    elif trial_pup.shape[0] < n_pupil_samples:
                        ei = trial_pup.shape[0]
                        results["att_pupil"]["raw"][ppi, ti, t_s, t_n, \
                            :ei] = trial_pup
                    else:
                        results["att_pupil"]["raw"][ppi, ti, t_s, t_n, :]\
                            = trial_pup[:n_pupil_samples]
                    
                    baseline_avg = numpy.nanmean(base_pup)
                    
                    if not numpy.isnan(baseline_avg):
                        results["att_pupil"]["corrected"][ppi, ti, t_s, t_n, :] = \
                            results["att_pupil"]["raw"][ppi, ti, t_s, t_n, :] \
                            - baseline_avg
                        results["att_pupil"]["corrected_baseline-divided"][ppi, ti, t_s, t_n, :] = \
                            results["att_pupil"]["raw"][ppi, ti, t_s, t_n, :] \
                            / baseline_avg
                    else:
                        print("\tOOPS, all NaNs! Trial {} for {}".format( \
                            i, exp_name))

                    results["att_pupil"]["reaction_time"][ppi, ti, t_s, t_n, t_t] =\
                        reaction_time
             
                
##### CALIBRATION averages
for tracker in trackers:
    
    fig, ax = pyplot.subplots()
    ax.set_ylim(DISPSIZE[1], 0)   
    ax.set_xlim(0, DISPSIZE[0])
    
    for px,py in CALIB_DOTS_PX:
        ax.plot(px, py, "o", markersize=5, \
            color="black")
    
        if tracker == "Alea":
            tracker_index = 0
        else:
            tracker_index = 1
        # Iterate over calibration phases
        for phase_index in range(results["calibration"]["x"].shape[2]):
            # Get x and y data for the current tracker and calibration phase
            x_data = results["calibration"]["x"][:, tracker_index, phase_index].flatten()
            y_data = results["calibration"]["y"][:, tracker_index, phase_index].flatten()
            
            if phase_index == 0:
                colour = "Green"
            else: 
                colour = "Orange"
            
            # Plot data for the current calibration phase
            ax.scatter(x_data, y_data, label ='Test point {}'.format(\
                phase_index +1), alpha = 0.1, s = 1, c = colour)
        
        # Set labels and title
        ax.set_xlabel('X coordinates')
        ax.set_ylabel('Y coordinates')
        ax.set_title('Fixation for {}'.format(tracker))
        fig.savefig(os.path.join(CALDIR,"fixation_all_{}.png".format( \
                tracker)))
        pyplot.close()  
      
    

    
# Compute average RMS across all participants.
rms_m = numpy.nanmean(results["calibration"]["rms_ang"], axis=0)
rms_sd = numpy.nanstd(results["calibration"]["rms_ang"], axis=0)
# Count the number of results that is not a NaN.
rms_n = numpy.sum(numpy.invert(numpy.isnan(results["calibration"]["rms"])), \
    axis=0)
# Compute the standard error of the mean.
rms_sem = rms_sd / numpy.sqrt(rms_n)

# Compute average accuracy across all participants.
acc_m = numpy.nanmean(results["calibration"]["acc_ang"], axis=0)
acc_sd = numpy.nanstd(results["calibration"]["acc_ang"], axis=0)
# Count the number of results that is not a NaN.
acc_n = numpy.sum(numpy.invert(numpy.isnan(results["calibration"]["acc_ang"])), \
    axis=0)
# Compute the standard error of the mean.
acc_sem = acc_sd / numpy.sqrt(acc_n)

# Compute average STD across all participants.
std_m = numpy.nanmean(results["calibration"]["std_ang"], axis=0)
std_sd = numpy.nanstd(results["calibration"]["std_ang"], axis=0)
# Count the number of results that is not a NaN.
std_n = numpy.sum(numpy.invert(numpy.isnan(results["calibration"]["std_ang"])), \
    axis=0)
# Compute the standard error of the mean.
std_sem = std_sd / numpy.sqrt(std_n)

# Create a DataFrame for accuracy results
acc_columns = ["Tracker", "Calibration Phase", "Target Number", \
    "Mean Accuracy", "SD Accuracy", "SEM Accuracy", "Number of Participants"]
acc_df = pandas.DataFrame(columns=acc_columns)

# Create a DataFrame for RMS results
rms_columns = ["Tracker", "Calibration Phase", "Target Number", \
    "Mean RMS", "SD RMS", "SEM RMS", "Number of Participants"]
rms_df = pandas.DataFrame(columns=rms_columns)

# STD results DataFrame
std_columns = ["Tracker", "Calibration Phase", "Target Number", \
    "Mean STD", "SD STD", "SEM STD", "Number of Participants"]
std_df = pandas.DataFrame(columns=std_columns)

# Loop through eye trackers and calibration phases
for ti, tracker in enumerate(trackers):
    for i_phase, calibration_phase in enumerate(calib_phases):
        for target_nr in range(n_calib_points):
            # Extract accuracy results
            acc_mean = acc_m[ti, i_phase, target_nr]
            acc_sd_value = acc_sd[ti, i_phase, target_nr]
            acc_sem_value = acc_sem[ti, i_phase, target_nr]
            acc_count = acc_n[ti, i_phase, target_nr]

            # Append accuracy results to the DataFrame
            acc_df = pandas.concat([acc_df, pandas.DataFrame({
                "Tracker": [tracker],
                "Calibration Phase": [calibration_phase],
                "Target Number": [target_nr],
                "Mean Accuracy": [acc_mean],
                "SD Accuracy": [acc_sd_value],
                "SEM Accuracy": [acc_sem_value],
                "Number of Participants": [acc_count]
            })], ignore_index=True)

            # Extract RMS results
            rms_mean = rms_m[ti, i_phase, target_nr]
            rms_sd_value = rms_sd[ti, i_phase, target_nr]
            rms_sem_value = rms_sem[ti, i_phase, target_nr]
            rms_count = rms_n[ti, i_phase, target_nr]

            # Append RMS results to the DataFrame
            rms_df = pandas.concat([rms_df, pandas.DataFrame({
                "Tracker": [tracker],
                "Calibration Phase": [calibration_phase],
                "Target Number": [target_nr],
                "Mean RMS": [rms_mean],
                "SD RMS": [rms_sd_value],
                "SEM RMS": [rms_sem_value],
                "Number of Participants": [rms_count]
            })], ignore_index=True)
            
            # Extract STD results 
            std_mean = std_m[ti, i_phase, target_nr]
            std_sd_value = std_sd[ti, i_phase, target_nr]
            std_sem_value = std_sem[ti, i_phase, target_nr]
            std_count = std_n[ti, i_phase, target_nr]
            
            # Append STD results to dataframe
            std_df = pandas.concat([std_df, pandas.DataFrame({
                "Tracker": [tracker],
                "Calibration Phase": [calibration_phase],
                "Target Number": [target_nr],
                "Mean STD": [std_mean],
                "SD STD": [std_sd_value],
                "SEM STD": [std_sem_value],
                "Number of Participants": [std_count]
            })], ignore_index=True)

# Save DataFrames to CSV files
acc_df.to_csv(os.path.join(CALDIR, "accuracy_results.csv"), index=False)
rms_df.to_csv(os.path.join(CALDIR, "rms_results.csv"), index=False)
std_df.to_csv(os.path.join(CALDIR, "std_results.csv"), index=False)


##### AOI averages 

#### all participants ####

# Initialise dict for mean and standard deviation for sample nr
prop_p_m = {}
prop_p_sd = {}

# Compute average fix_nr across all participants. #
# Average across trackers and Images
for aoi_name in aoi_features:
    prop_p_m[aoi_name] = numpy.nanmean(results["aoi"]["proportion"][aoi_name], axis=0)
    prop_p_sd[aoi_name] = numpy.nanstd(results["aoi"]["proportion"][aoi_name], axis=0)

#initialise dict for mean and standard deviation for dwell

dwell_p_m = {}
dwell_p_sd = {}


# Compute average dwell across all participants.
for aoi_name in aoi_features:
    dwell_p_m[aoi_name] = numpy.nanmean(results["aoi"]["dwell"][aoi_name], axis=0)
    dwell_p_sd[aoi_name] = numpy.nanstd(results["aoi"]["dwell"][aoi_name], axis=0)    

# Create a DataFrame for proportion results.
prop_p_columns = ["Tracker", "Image Trial", \
    "Mean Sample number", "SD Sample number"]
prop_p_df = pandas.DataFrame(columns=prop_p_columns)

# Create a DataFrame for dwell time results
dwell_p_columns = ["Tracker", "Image Trial", \
    "Mean dwell", "SD dwell"]
dwell_p_df = pandas.DataFrame(columns=dwell_p_columns)

# Loop through eye trackers and aoi image
for ti, tracker in enumerate(trackers):
    for img_name in image_names:
        if img_name == 'CFD-LF':
            img_index = 0
        else:
            img_index = 1
        for aoi_name in aoi_features:
            # Extract sample number results
            prop_p_mean = prop_p_m[aoi_name][ti, img_index]
            prop_p_sd_value = prop_p_sd[aoi_name][ti, img_index]
            # Append proportion results to the DataFrame
            prop_p_df = pandas.concat([prop_p_df, pandas.DataFrame({
                "Tracker": [tracker],
                "Image Trial": [img_name],
                "AOI": [aoi_name],
                "Mean Sample number": [prop_p_mean],
                "SD Sample number": [prop_p_sd_value],
            })], ignore_index=True)

            # Extract Dwell results
            dwell_p_mean = dwell_p_m[aoi_name][ti, img_index]
            dwell_p_sd_value = dwell_p_sd[aoi_name][ti, img_index]
      
            # Append dwell results to the DataFrame
            dwell_p_df = pandas.concat([dwell_p_df, pandas.DataFrame({
                "Tracker": [tracker],
                "Image Trial": [img_name],
                "AOI": [aoi_name],
                "Mean dwell": [dwell_p_mean],
                "SD dwell": [dwell_p_sd_value],
            })], ignore_index=True)
            
# Save DataFrames to CSV files
prop_p_df.to_csv(os.path.join(AOIDIR, "proportion_p_results.csv"), index=False)
dwell_p_df.to_csv(os.path.join(AOIDIR, "dwell_p_results.csv"), index=False)

# Averaged aoi plot per image
fig = {}
ax = {}
for tracker in trackers:
    for aoi_image_name in image_names:
        # Add new figure to the dicts.
        fig_w = 8.0
        fig_wh_ratio = DISPSIZE[0] / DISPSIZE[1]
        fig_h = fig_w / fig_wh_ratio
        fig[aoi_image_name] = pyplot.Figure(figsize=(fig_w, fig_h))
        ax[aoi_image_name] = fig[aoi_image_name].add_axes([0, 0, 1, 1])

        # Load the background image
        fpath_img = os.path.join(IMGDIR, aoi_image_name) + ".jpg"
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
        
        # Setting the limits of the plot to the screen size
        ax[aoi_image_name].imshow(frame, \
            extent=[x_si, x_ei, y_ei, y_si])
        ax[aoi_image_name].set_ylim(DISPSIZE[1], 0)   
        ax[aoi_image_name].set_xlim(0, DISPSIZE[0])
        
        if tracker == "Alea":
            tracker_index = 0
        else:
            tracker_index = 1
            
        # index image
        if aoi_image_name == "CFD-LF": 
            img_index = 0
        else:
            img_index = 1    
        
        # Get x and y data for the current tracker and calibration phase
        x_data = results["aoi"]["x"][:, tracker_index, img_index,:].flatten()
        y_data = results["aoi"]["y"][:, tracker_index, img_index,:].flatten()
        
        # filter nan values
        x_data = x_data[~numpy.isnan(x_data)]
        y_data = y_data[~numpy.isnan(y_data)]
        
        # Plot heatmap of all samples
        if tracker_index == 0:    
            heatmap, xedges, yedges = numpy.histogram2d(x_data, y_data, bins=1100)
        else:
            heatmap, xedges, yedges = numpy.histogram2d(x_data, y_data, bins=800)

        
        # Apply Gaussian filter for smoothing
        heatmap_smooth = gaussian_filter(heatmap.T, sigma=24)
        
        # Normalize the histogram
        max_count = numpy.max(heatmap_smooth)
        heatmap_smooth /= max_count
        
        # Set bins with zero counts to transparent
        heatmap_smooth[numpy.where(heatmap_smooth < 0.04)] = numpy.nan
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        # Plot heatmap
        ax[aoi_image_name].imshow(heatmap_smooth, extent=extent, origin='lower', cmap=cm.jet, alpha=0.5)
        
        # Set labels and title
        ax[aoi_image_name].set_xlabel('X coordinates')
        ax[aoi_image_name].set_ylabel('Y coordinates')
        ax[aoi_image_name].set_title('Heat map plot for {}_{}'.format(tracker,aoi_image_name))
          
        # Save and close plots
        fig[aoi_image_name].savefig(os.path.join(AOIDIR,"Heat_plot_{}_{}.png".format( \
                tracker, aoi_image_name)))
        pyplot.close(fig[aoi_image_name])  
       

##### Pupil averages 

#initialise dict for mean and standard deviation for corrected pup
pup_p_m = {}
pup_p_sd = {}

# Compute average pupil size across all participants. 
pup_p_m = \
    numpy.nanmean(results["pupil"]["corrected"], axis=0)
pup_p_sd = \
    numpy.nanstd(results["pupil"]["corrected"], axis=0)

# Count the number of results that is not a NaN.
pup_n = numpy.sum(numpy.invert(numpy.isnan(results["pupil"]["corrected"])), \
    axis=0)
# Compute the standard error of the mean.
pup_sem = pup_p_sd / numpy.sqrt(pup_n)

#initialise dict for mean and standard deviation for corrected pup over trials
pup_p_tn_m = {}
pup_p_tn_sd = {}

# Compute average pupil size across all participants and trials
pup_p_tn_m = numpy.nanmean(pup_p_m, axis= 2)
pup_p_tn_sd = numpy.nanmean(pup_p_sd, axis= 2)

# Plot the averaged 
for i in range(pup_p_tn_m.shape[0]):  # Iterate over the trackers
    for j in range(pup_p_tn_m.shape[1]):  # Iterate over the conditions
        if i == 0:
            ti = 'Alea'
        else:
            ti = 'EyeLink'
        if j == 0:
            scr = 'Dark'
        else:
            scr = 'Light'
        # Plot data
        pyplot.plot(pup_p_tn_m[i, j, :],label=('{}'.format(scr)))
        mean = pup_p_tn_m[i, j, :]
        std = pup_p_tn_sd[i, j, :]
        pyplot.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

    pyplot.xlabel('Samples over time')
    pyplot.ylabel('Corrected pupil size')
    if i == 0:
        pyplot.title('Corrected Pupil size plot for Alea')
    else:
        pyplot.title('Corrected Pupil size plot for EyeLink')
    pyplot.grid(True)
    pyplot.legend()
    
    # Save and Close
    pyplot.savefig(os.path.join(PUPDIR,"Pupil_average_{}.png"\
                                .format(ti)))
    pyplot.clf()



###### ATT pupil averages 

#initialise dict for mean and standard deviation for corrected pup
att_p_m = {}
att_p_sd = {}

# Compute average pupil size across all participants. 
att_p_m = \
    numpy.nanmean(results["att_pupil"]["corrected"], axis=0)
att_p_sd = \
    numpy.nanstd(results["att_pupil"]["corrected"], axis=0)

# Count the number of results that is not a NaN.
att_n = numpy.sum(numpy.invert(numpy.isnan(results["att_pupil"]["corrected"])), \
    axis=0)
# Compute the standard error of the mean.
att_sem = att_p_sd / numpy.sqrt(att_n)

#initialise dict for mean and standard deviation for corrected pup over trials
att_p_tn_m = {}
att_p_tn_sd = {}

att_p_tn_m = numpy.nanmean(att_p_m, axis= 2)
att_p_tn_sd = numpy.nanmean(att_p_sd, axis= 2)

# Plot the averaged 
val = numpy.copy(results["att_pupil"]["corrected_baseline-divided"])
val[:,0,:,:,n_pupil_samples_alea:] = numpy.nan
val = numpy.nanmean(val, axis=3)
for i in range(att_p_tn_m.shape[0]):  # Iterate over the first dimension
    m = numpy.nanmean(val[:,i,:,:], axis=0)
    subject_m = numpy.nanmean(numpy.nanmean(val[:,i,:,:], axis=2), axis=1).reshape(-1,1,1)
    grand_m = numpy.nanmean(subject_m)
    nv = val[:,i,:,:] - subject_m + grand_m
    ci95 = 1.96 * (numpy.nanstd(nv, axis=0, ddof=1) / numpy.sqrt(n_participants))
    for j in range(att_p_tn_m.shape[1]):  # Iterate over the second dimension
        
        # set the sample length and condition
        if i == 0:
            ti = 'Alea'
            ei = n_pupil_samples_alea
        else:
            ti = 'EyeLink'
            ei = n_pupil_samples
        if j == 0:
            scr = 'Attend Dark'
        else:
            scr = 'Attend Light'
        
        plot_time = numpy.linspace(0, 2.5, ei)
        pyplot.plot(plot_time, m[j, :ei],label=('{}'.format(scr)))
        pyplot.fill_between(plot_time, \
            m[j,:ei] - ci95[j,:ei], \
            m[j,:ei] + ci95[j,:ei], alpha=0.2)

    pyplot.xlabel('Time (sec)')
    pyplot.ylabel('Corrected pupil size')
    if i == 0:
        pyplot.title('Corrected Pupil size plot for Alea')
    else:
        pyplot.title('Corrected Pupil size plot for EyeLink')
    pyplot.grid(True)
    pyplot.legend()
    # save and close plots
    pyplot.savefig(os.path.join(ATTDIR,"ATT_pupil_average_{}.png"\
                                .format(ti)))
    pyplot.close()


#### STATS TESTS ######

### Calibration analysis

# Loop through all outcome measures (acc, rms, std, acc_ang, rms_ang, std_ang)
bf01_threshold = 3
p_threshold = 0.05

for measure in calib_measures_analyse:
       
    betas = numpy.zeros((3,n_calib_points), dtype=numpy.float64)
    t = numpy.zeros((3,n_calib_points), dtype=numpy.float64)
    p = numpy.zeros((3,n_calib_points), dtype=numpy.float64)
    bf01 = numpy.zeros((3,n_calib_points), dtype=numpy.float64)
    t_posthoc = numpy.zeros((2,n_calib_points), dtype=numpy.float64)
    p_posthoc = numpy.zeros((2,n_calib_points), dtype=numpy.float64)
    bf01_posthoc = numpy.zeros((2,n_calib_points), dtype=numpy.float64)
    
    # Loop through all points.
    for i in range(n_calib_points):
    
        endog = results["calibration"][measure][:,:,:,i]
        
        # Construct exogenous matrices that code for group (participant number),
        # tracker, and phase.
        ppnames_array = numpy.array(ppnames)
        group = numpy.zeros(endog.shape, dtype=ppnames_array.dtype)
        for ppname in ppnames:
            sel = ppnames_array == ppname
            group[sel,:,:] = ppname
        tracker = numpy.zeros(endog.shape, dtype=numpy.int64)
        tracker[:,0,:] = 0
        tracker[:,1,:] = 1
        phase = numpy.zeros(endog.shape, dtype=numpy.int64)
        phase[:,:,0] = 0
        phase[:,:,1] = 1
        
        
        endog = endog.flatten()
        endog = (endog - numpy.nanmean(endog))/numpy.nanstd(endog)
        exog = numpy.ones((endog.shape[0], 3), dtype=numpy.float64)
        exog[:,1] = tracker.flatten()
        exog[:,2] = phase.flatten()
        group = group.flatten()
        include = numpy.invert(numpy.isnan(endog))
        
        model = MixedLM(endog[include], exog[include,:], group[include], \
            missing="none")
        model_fit = model.fit()
        betas[:,i] = model_fit.params[:betas.shape[0]]
        # Compute t statistics for the betas.
        t[:,i] = betas[:,i] / model_fit.bse[:betas.shape[0]]
        p[:,i] = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t[:,i]), \
            n_participants-1))
        # Compute Bayes Factor for the tracker parameter t value.
        for j in range(t.shape[0]):
            bf01[j,i] = 1.0 / pingouin.bayesfactor_ttest(t[j,i], \
                n_participants, paired=True, alternative="two-sided", r=0.707)
        # Post-hoc t-test to compare pre and post measurements within each
        # tracker type.
        for j, tracker_type in enumerate(trackers):
            hoc_results = results["calibration"][measure][:,:,:,i]
            hoc_results[:,:,:] = (hoc_results[:,:,:] - numpy.nanmean\
                (hoc_results[:,:,:]))/numpy.nanstd(hoc_results[:,:,:])

            pre_ = hoc_results[:,j,0]
            post_ = hoc_results[:,j,1]  
 
            t_posthoc[j,i], p_posthoc[j,i] = scipy.stats.ttest_rel( \
                pre_, post_, nan_policy="omit")
            bf01_posthoc[j,i] = 1.0 / pingouin.bayesfactor_ttest( \
                t_posthoc[j,i], pre_.shape[0], paired=True, \
                alternative="two-sided", r=0.707)

    # Create results table
    fpath = os.path.join(CALDIR, "stats_for_calibration_{}.csv".format(measure))
    with open(fpath, "w") as f:
        header = ["sample"]
        for predictor in ["intercept", "tracker", "phase"]:
            for var in ["beta", "t", "p", "BF01"]:
                header.append("{}_{}".format(predictor, var))
        for tracker in trackers:
            for var in ["t", "p", "BF01"]:
                header.append("{}_post-hoc_{}".format(var, tracker))
        f.write(",".join(header))
    
        for i in range(betas.shape[1]):
            line = [i]
            for j in range(betas.shape[0]):
                for var in [betas, t, p, bf01]:
                    line.append(var[j,i])
            for j, tracker_type in enumerate(trackers):
                for var in [t_posthoc, p_posthoc, bf01_posthoc]:
                    line.append(var[j,i])
            f.write("\n" + ",".join(map(str, line)))
    
    # Create figure for the stats.
    beta_names = ["constant", "tracker", "phase"]
    fig, ax = pyplot.subplots()
    for bi in range(betas.shape[0]):
        ax.plot(betas[bi,:], "-", label=beta_names[bi])
    ax.legend()
    fig.savefig(os.path.join(CALDIR, "betas_for_calibration_{}.png".format(measure)))
    pyplot.close(fig)
    
    # Create a figure for bf01 and p-values
    bf01_evidence = numpy.zeros(results["calibration"][measure].shape[3])
    bf01_evidence.fill(bf01_threshold)
    bf01_names = ["constant", "tracker", "phase"]
    fig, ax = pyplot.subplots()
    plot_target = numpy.linspace(1, 12, 12)
    for bi in range(bf01.shape[0]):
        ax.plot(plot_target, bf01[bi,:], "-", label=bf01_names[bi])
    ax.plot(plot_target, bf01_evidence, linestyle='dashed', color = 'gray')
    ax.fill_between(plot_target,bf01_evidence , color='grey', alpha=0.5)
    ax.set_xticks(numpy.arange(1, 13, 1))
    ax.set_xlabel("Target_number")
    ax.set_ylabel("BF_01")
    ax.legend()
    fig.savefig(os.path.join(CALDIR, "bf01_for_calibration_{}.png".format(measure)))
    pyplot.close(fig)
    
    # Create a p-values plot
    p_evidence = numpy.zeros(results["calibration"][measure].shape[3])
    p_evidence.fill(p_threshold)
    p_names = ["constant", "tracker", "phase"]
    fig, ax = pyplot.subplots()
    plot_target = numpy.linspace(1, 12, 12)
    for bi in range(p.shape[0]):
        ax.plot(plot_target, p[bi,:], "-", label=p_names[bi])
    ax.plot(plot_target, p_evidence, linestyle='dashed', color = 'gray')
    ax.fill_between(plot_target,p_evidence , color='grey', alpha=0.5)
    ax.set_xticks(numpy.arange(1, 13, 1))
    ax.set_xlabel("Target_number")
    ax.set_ylabel("p_value")
    ax.legend()
    fig.savefig(os.path.join(CALDIR, "p_for_calibration_{}.png".format(measure)))
    pyplot.close(fig)
        
        
### AOI analysis - LME

# Do one LME per outcome variable. Will be focusing on dwell and 
# p(sample), but it's easier to write the generic format out.
for img, img_name in enumerate(image_names):
    for var in ["fix_nr", "dwell", "proportion"]: # results["aoi"].keys():
        # Create a long-format CSV file with the current variable's computed
        # results in. We'll need this for the LME.
        file_path = os.path.join(AOIDIR, "aoi-results_{}_{}_long.csv".format( \
            img_name, var))
        with open(file_path, "w") as f:
            header = ["ppname", "tracker", "aoi", var]
            f.write(",".join(header))
            aoi_names = list(results["aoi"][var].keys())
            aoi_names.remove("other")
            if img_name == 'CFD-LF':
                aoi_names.remove("ear")
            for ppi, ppname in enumerate(ppnames):
                for ti, tracker_name in enumerate(trackers):
                    for aoi in aoi_names:
                        line = [ppname, tracker_name, \
                            aoi.replace("face", "aface"), \
                            results["aoi"][var][aoi][ppi,ti,img]]
                        f.write("\n" + ",".join(map(str, line)))
        # Load the data just saved into a pandas dataframe.
        data = pandas.read_csv(file_path)
        n_original = len(numpy.unique(data["ppname"]))
        if var in data.keys():
            m = numpy.nanmean(data[var])
            sd = numpy.nanstd(data[var])
            data[var] = (data[var] - m) / sd
        # Fit the current model.
        formulae = [ \
            "{} ~ tracker + aoi + tracker*aoi".format(var), \
            "{} ~ tracker + aoi".format(var), \
            "{} ~ tracker".format(var), \
            "{} ~ aoi".format(var), \
            "{} ~ 1".format(var), \
            ]
        fpath = os.path.join(AOIDIR, "aoi-lme_{}_{}.txt".format(img_name, var))
        with open(fpath, "w") as f:
            for fi, formula in enumerate(formulae):
                # Fit the curren model.
                lme = MixedLM.from_formula(formula, groups=data["ppname"], \
                    data=data, missing="drop")
                lme = lme.fit()
                # Write outcomes to file.
                f.write(lme.summary().as_text())
                f.write(formula)
                f.write("\nAIC = {}; BIC = {}".format( \
                    aic(lme.llf, lme.nobs, lme.df_modelwc), \
                    bic(lme.llf, lme.nobs, lme.df_modelwc)))
                f.write("\n\n\n")
                # Write a file just for this model.
                fpath_ = os.path.join(AOIDIR, "aoi-lme-{}_{}_{}.txt".format( \
                    img_name, var, fi))
                with open(fpath_, "w") as f_:
                    header = ["param", "beta", "se", "t", "p", "BF10", "BF01", \
                        "CI95_lo", "CI95_hi", "report"]
                    f_.write("\t".join(map(str, header)))
                    # Compute t values, p values, and Bayes Factors.
                    # Get the beta estimates.
                    beta = lme.params
                    # Get the parameter names in the beta dict.
                    beta_names = beta.keys()
                    # Compute confidence intervals.
                    ci_95 = lme.conf_int(alpha=0.05)
                    # Grab the standard errors.
                    se = lme.bse
                    # Compute t values.
                    t = beta / se
                    # Write all parameters to file.
                    for param in beta_names:
                        
                        # Skip the group variance param.
                        if param == "Group Var":
                            continue
                        
                        # Compute a p value for the t test.
                        p = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs( \
                            t[param]), n_original-1))
                        # Compute a Bayes Factor (01) for the t-test.
                        bf10 = pingouin.bayesfactor_ttest(t[param], \
                            n_original, paired=True, alternative="two-sided", \
                            r=0.707)
                        
                        # Write a report for this test.
                        report_val = [ \
                            numpy.format_float_scientific(beta[param], precision=2), \
                            numpy.format_float_scientific(se[param], precision=2), \
                            numpy.format_float_scientific(ci_95[0][param], precision=2), \
                            numpy.format_float_scientific(ci_95[1][param], precision=2), \
                            n_original-1, \
                            numpy.format_float_scientific(t[param], precision=2), \
                            numpy.format_float_scientific(p, precision=3), \
                            ]
                        report_txt = "\"beta={} (se={}, 95% CI=[{},{}]), t({})={}, p={}"
                        if p < 0.05:
                            report_val.append(numpy.format_float_scientific( \
                                bf10, precision=2))
                            report_txt += ", BF10={}\""
                        else:
                            report_val.append(numpy.format_float_scientific( \
                                1.0/bf10, precision=2))
                            report_txt += ", BF01={}\""
                        
                        # Write to file.
                        line = [param, beta[param], se[param], t[param], p, \
                            bf10, 1.0/bf10, ci_95[0][param], ci_95[1][param], \
                            report_txt.format(*report_val)]
                        f_.write("\n" + "\t".join(map(str, line)))

### pupil analysis


for exp_name in ["pupil", "att_pupil"]:
    if exp_name == "pupil":
        outputdir = PUPDIR
    else:
        outputdir = ATTDIR
    
    # Downsample the EyeLink data.
    n_pupil_samples_alea = int(PUPTRIALTIME / ALEA_SAMPLE_MS)
    shape = list(results[exp_name]["corrected_baseline-divided"].shape)
    shape[-1] = n_pupil_samples_alea
    pup_downsampled = numpy.zeros(shape, \
        dtype=results[exp_name]["corrected_baseline-divided"].dtype)
    for ti, tracker in enumerate(trackers):
        if tracker == "Alea":
            pup_downsampled[:,ti,:,:,:] = \
                results[exp_name]["corrected_baseline-divided"][:,ti,:,:,:n_pupil_samples_alea]
        else:
            selected_indices = numpy.round(numpy.linspace(0, n_pupil_samples-1, \
                n_pupil_samples_alea)).astype(numpy.int64)
            pup_downsampled[:,ti,:,:,:] = \
                results[exp_name]["corrected_baseline-divided"][:,:,:,:,selected_indices][:,ti,:,:,:]
        
    # Standardise within each tracker.
    for ti, tracker_name in enumerate(trackers):
        m = numpy.nanmean(pup_downsampled[:,ti,:,:,:])
        sd = numpy.nanstd(pup_downsampled[:,ti,:,:,:])
        pup_downsampled[:,ti,:,:,:] = (pup_downsampled[:,ti,:,:,:]-m)/sd
    
    # Loop through both trackers, and the combined trackers.
    for test_set in ["combined", "Alea", "EyeLink"]:
        
        if test_set != "combined":
            ti = trackers.index(test_set)
            n_betas = 2
        else:
            n_betas = 3
    
        # Loop through all samples.
        betas = numpy.zeros((n_betas,n_pupil_samples_alea), dtype=numpy.float64)
        t = numpy.zeros((n_betas,n_pupil_samples_alea), dtype=numpy.float64)
        p = numpy.zeros((n_betas,n_pupil_samples_alea), dtype=numpy.float64)
        bf01 = numpy.zeros((n_betas,n_pupil_samples_alea), dtype=numpy.float64)
        for i in range(n_pupil_samples_alea):
            
            # Get the "endogenous" (dependent) value, which is pupil size.
            if test_set == "combined":
                endog = pup_downsampled[:,:,:,:,i]
            else:
                endog = pup_downsampled[:,ti,:,:,i]
            # Construct exogenous matrices that code for group (participant number),
            # tracker, and condition (light/dark screen).
            ppnames_array = numpy.array(ppnames)
            group = numpy.zeros(endog.shape, dtype=ppnames_array.dtype)
            for ppname in ppnames:
                sel = ppnames_array == ppname
                if test_set == "combined":
                    group[sel,:,:,:] = ppname
                else:
                    group[sel,:,:] = ppname
            if test_set == "combined":
                tracker = numpy.zeros(endog.shape, dtype=numpy.int64)
                tracker[:,0,:,:] = 0
                tracker[:,1,:,:] = 1
                attend_light = numpy.zeros(endog.shape, dtype=numpy.int64)
                attend_light[:,:,0,:] = 0
                attend_light[:,:,1,:] = 1
            else:
                attend_light = numpy.zeros(endog.shape, dtype=numpy.int64)
                attend_light[:,0,:] = 0
                attend_light[:,1,:] = 1

            # Flatten all matrices so that we can use them in the linear model.
            endog = endog.flatten()
            if test_set == "combined":
                exog = numpy.ones((endog.shape[0], 3), dtype=numpy.float64)
                exog[:,1] = tracker.flatten()
                exog[:,2] = attend_light.flatten()
            else:
                exog = numpy.ones((endog.shape[0], 2), dtype=numpy.float64)
                exog[:,1] = attend_light.flatten()
            group = group.flatten()
            # Determine which results to include (all not-NaN).
            include = numpy.invert(numpy.isnan(endog))
            # Run the model.
            model = MixedLM(endog[include], exog[include,:], group[include], \
                missing="none")
            model_fit = model.fit()
            betas[:,i] = model_fit.params[:betas.shape[0]]
            # Compute t statistics for the betas.
            t[:,i] = betas[:,i] / model_fit.bse[:betas.shape[0]]
            p[:,i] = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs(t[:,i]), \
                n_participants-1))
            # Compute Bayes Factor for the tracker parameter t value.
            for j in range(t.shape[0]):
                bf01[j,i] = 1.0 / pingouin.bayesfactor_ttest(t[j,i], \
                    n_participants, paired=True, alternative="two-sided", r=0.707)
                    
        # Create results table
        fpath = os.path.join(outputdir, "stats_for_{}_{}.csv".format(exp_name, test_set))
        with open(fpath, "w") as f:
            header = ["sample"]
            if test_set == "combined":
                predictors = ["intercept", "tracker", "attend"]
            else:
                predictors = ["intercept", "attend"]
            for predictor in predictors:
                for var in ["beta", "t", "p", "BF01"]:
                    header.append("{}_{}".format(predictor, var))
            f.write(",".join(header))
    
            for i in range(betas.shape[1]):
                line = [i]
                for j in range(betas.shape[0]):
                    for var in [betas, t, p, bf01]:
                        line.append(var[j,i])
                f.write("\n" + ",".join(map(str, line)))
        
        # Create figure for the stats.
        if test_set == "combined":
            beta_names = ["constant", "tracker", "attend_light"]
        else:
            beta_names = ["constant", "attend_light"]            
        fig, ax = pyplot.subplots()
        for bi in range(betas.shape[0]):
            ax.plot(betas[bi,:], "-", label=beta_names[bi])
        ax.legend()
        fig.savefig(os.path.join(outputdir, "betas_for_{}_{}.png".format( \
            exp_name, test_set)))
        pyplot.close(fig)


# Excluded samples 
# Dict for number of excluded trials in each experiment for the trackers

shape = (n_trackers)
excluded = {}
for var in ['calibration', 'aoi', 'pupil', 'att_pupil']:
    excluded[var] = numpy.zeros(shape, dtype=numpy.float64) * numpy.nan

# Calibration 
for ti, tracker in enumerate(trackers):   
        
    if ti == 0: 
        cal_samples = n_calibration_samples_alea
        aoi_sample = n_image_samples_alea
        pup_sample = n_pupil_samples_alea
    else: 
        cal_samples = CALIB_DOT_DUR
        aoi_sample = IMGTIME
        pup_sample = PUPTRIALTIME
     
    # calibration    
    valid = numpy.sum(numpy.invert(numpy.isnan(results['calibration']['x'][:,ti,:,:,:])))
    total_samples = n_participants * n_calib_phases * n_calib_points * cal_samples
    excluded['calibration'][ti,] = total_samples - valid
    
    # aoi
    valid = numpy.sum(numpy.invert(numpy.isnan(results['aoi']['x'][:,ti,:,:])))
    total_samples = n_participants * n_images * aoi_sample
    excluded['aoi'][ti,] = total_samples - valid

    # pup
    valid = numpy.sum(numpy.invert(numpy.isnan(results["pupil"]["raw"][:,ti,:,:,:])))
    total_samples = n_participants * n_attend_cond * n_trials * pup_sample
    excluded['pupil'][ti,] = total_samples - valid
    
    # att_pup
    valid = numpy.sum(numpy.invert(numpy.isnan(results["att_pupil"]["raw"][:,ti,:,:,:])))
    total_samples = n_participants * n_attend_cond * n_trials * pup_sample
    excluded['att_pupil'][ti,] = total_samples - valid
