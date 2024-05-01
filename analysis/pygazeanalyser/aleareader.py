# EDF Reader
#
# Does not actually read EDFs directly, but the ASC files that are produced
# by edf2asc (SR Research). Information on saccades, fixations and blinks is
# read from the EDF, therefore based on SR Research algorithms. For optimal
# event detection, it might be better to use a different algorithm, e.g.
# Nystrom, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation,
# saccade, and glissade detection in eyetracking data. Behavior Research
# Methods, 42, 188-204. doi:10.3758/BRM.42.1.188
#
# (C) Edwin Dalmaijer, 2013-2014
# edwin.dalmaijer@gmail.com
#
# version 2 (24-Apr-2014)

__author__ = "Edwin Dalmaijer"


import copy
import os.path

import numpy


def replace_missing(value, missing=0.0):
    
    """Returns missing code if passed value is missing, or the passed value
    if it is not missing; a missing value in the EDF contains only a
    period, no numbers; NOTE: this function is for gaze position values
    only, NOT for pupil size, as missing pupil size data is coded '0.0'
    
    arguments
    value        -    either an X or a Y gaze position value (NOT pupil
                    size! This is coded '0.0')
    
    keyword arguments
    missing        -    the missing code to replace missing data with
                    (default = 0.0)
    
    returns
    value        -    either a missing code, or a float value of the
                    gaze position
    """
    
    if value.replace(' ','') == '.':
        return missing
    else:
        return float(value)

def read_alea(filename, start, stop=None, missing=0.0, debug=False):
    
    """Returns a list with dicts for every trial. A trial dict contains the
    following keys:
        x        -    numpy array of x positions
        y        -    numpy array of y positions
        size        -    numpy array of pupil size
        time        -    numpy array of timestamps, t=0 at trialstart
        trackertime    -    numpy array of timestamps, according to EDF
        events    -    dict with the following keys:
                        Sfix    -    list of lists, each containing [starttime]
                        Ssac    -    list of lists, each containing [starttime]
                        Sblk    -    list of lists, each containing [starttime]
                        Efix    -    list of lists, each containing [starttime, endtime, duration, endx, endy]
                        Esac    -    list of lists, each containing [starttime, endtime, duration, startx, starty, endx, endy]
                        Eblk    -    list of lists, each containing [starttime, endtime, duration]
                        msg    -    list of lists, each containing [time, message]
                        NOTE: timing is in tracker time!
    
    arguments
    filename        -    path to the file that has to be read
    start        -    trial start string
    
    keyword arguments
    stop            -    trial ending string (default = None)
    missing        -    value to be used for missing data (default = -1)
    debug        -    Boolean indicating if DEBUG mode should be on or off;
                if DEBUG mode is on, information on what the script
                currently is doing will be printed to the console
                (default = False)
    
    returns
    data            -    a list with a dict for every trial (see above)
    """

    # # # # #
    # debug mode
    
    if debug:
        def message(msg):
            print(msg)
    else:
        def message(msg):
            pass
        
    
    # Check if the file exists
    if os.path.isfile(filename):
        # Open file, and read its contents.
        message("Reading file {}".format(filename))
        with open(filename, "r") as f:
            raw = f.readlines()
        message("Found {} lines".format(len(raw)))
            
    # raise exception if the file does not exist
    else:
        raise Exception("Error in read_edf: file '{}' does not exist".format( \
            filename))
    
    # Extract header.
    header = raw.pop(0)
    # Cut off the final newline.
    header = header[:-1]
    # Split by tabs to get each individual variable name.
    header = header.split()
    # Get the indices for variables we need.
    i_type = header.index("TYPE")
    i_t = header.index("rawDataTimeStamp")
    i_msg = i_t + 1
    i_x = header.index("intelliGazeX")
    i_y = header.index("intelliGazeY")
    i_pup_l = header.index("pupilDiameterLeftEye")
    i_pup_r = header.index("pupilDiameterRightEye")

    
    # # # # #
    # Parse lines
    
    # variables
    data = []
    x = []
    y = []
    size = []
    time = []
    trackertime = []
    events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
    starttime = 0
    started = False
    trialend = False
    finalline = raw[-1]
    
    # loop through all lines
    for line in raw:
        
        # Split the line by tabs.
        line = line.split("\t")

        # check if trial has already started
        if started:
            # only check for stop if there is one
            if stop is not None:
                if stop in line[i_msg]:
                    started = False
                    trialend = True
            # check for new start otherwise
            else:
                if (start in line[i_msg]) or (line == finalline):
                    started = True
                    trialend = True
            
            # # # # #
            # trial ending
            
            if trialend:
                message("trialend {}; {} samples found".format(len(data), \
                    len(x)))
                # trial dict
                trial = {}
                trial['x'] = numpy.array(x)
                trial['y'] = numpy.array(y)
                trial['size'] = numpy.array(size)
                trial['time'] = numpy.array(time)
                trial['trackertime'] = numpy.array(trackertime)
                trial['events'] = copy.deepcopy(events)
                # add trial to data
                data.append(trial)
                # reset stuff
                x = []
                y = []
                size = []
                time = []
                trackertime = []
                events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
                trialend = False
                
        # check if the current line contains start message
        else:
            if start in line[i_msg]:
                message("trialstart {}".format(len(data)))
                # set started to True
                started = True
                # find starting time
                starttime = int(line[i_t])
        
        # # # # #
        # parse line
        
        # We only have two tupes of lines: MSG and DAT for messages and
        # data. We'll treat each accordingly.
        if started:
            
            # MSG lines contain three things: TYPE (=="MSG"), timestamp, and
            # the actual message.
            if line[i_type] == "MSG":
                t = int(line[i_t])
                m = line[i_msg]
                events['msg'].append([t,m])
            
            # DAT lines contain data, which is described by the header.
            if line[i_type] == "DAT":
                
                # Get the timestamp.
                t = int(line[i_t])

                # Get the gaze position.
                x_ = float(line[i_x])
                y_ = float(line[i_y])
                # Get the pupil size, and compute the combined one.
                pl = float(line[i_pup_l])
                pr = float(line[i_pup_r])
                
                # If both pupil values are 0, data is MISSING.
                if (pl == 0) & (pr == 0):
                    x_ = missing
                    y_ = missing
                    p = missing
                elif (pl == 0) & (pr > 0):
                    p = pr
                elif (pl > 0) & (pr == 0):
                    p = pl
                else:
                    p = (pl + pr) / 2.0
                
                # Store data.
                x.append(x_)
                y.append(y_)
                size.append(p)
                time.append(t-starttime)
                trackertime.append(t)
    
    
    # # # # #
    # return
    
    return data
