# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:13:13 2022

@author: jlab
"""
#%%

from slmRA_main import SLM_RA
import pyqtgraph as pg
import numpy as np
from slmRA_plots import ImagePlot
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
#%% create microscope object (connect to SLM, initialize parameters)
m = SLM_RA("C:\\Users\\jlab\\.spyder-py3\\RandomAccess", "Z:\\User\\Caro\\RA\\Data", hardware = True) 
                          
#%% run if SLM power is OFF (usually after POWERDOWN, otherwise should start autonmatically when turning on power supply)
#m.bud.poll('BOOTUP')
#%% check status of GLV again and set VDDAH parameter to 350 (maximum specified by manufacturer)
m.bud.status()
m.bud.vddah()
#%% initialize parameters for scanning
# central position of defocus on SLM
m.x_0 = 10
# central pos for y axis
m.y_0 = -60
# top edge of FOV
m.x_min = -42*4
# bottom edge of FOV
m.x_max = -2*4
# n pixels top to bottom
m.x_steps =121 #121
# the same for y axis
m.y_min = -125*4
m.y_max = -5*4
m.y_steps = 361 #361
# the same for z axis, use in range +250 to -250 or less
m.def_min = -20
m.def_max = +20
m.z_steps_full = 61
m.x_scale = (m.x_steps-1)/(m.x_max-m.x_min)
m.y_scale = (m.y_steps-1)/(m.y_max-m.y_min)

# wavelength in nm
m.wl = 940
# cyl lens focal length in m
m.f_cyl = 32e3
# calibrate between voltage and deflection (don't change)
m.calib_f = 690
m.calib_p = 4
m.maxpi = 1.535

m.corr1 = 0
m.corr_n = np.zeros(1088)

m.xpix = (m.x_max-m.x_min)/(4*m.x_steps)
m.ypix = (m.y_max-m.y_min)/(4*m.y_steps)

m.is_stack = False

#%% load correction pattern acquired for 40x objective
m.corr060723_1753_40x, m.path_40x = m.load_data()

#%% calculate all patterns specified by parameters (can be arbitrary number)
m.calculate_stacks(corr = m.corr060723_1753_40x)

#%% send subset of planes in stack starting at plane z1, n1 = number of planes
z1 = 30#starts at 0, max: m.z_steps_full-1
n1 = 1
m.send_patterns_n(z = z1, n = n1)

#%% start preview acquisition (data is not saved)
m.syncRate = 50e3
m.sampleRate =2e6
scale =10e2

m.preview(scale, False)

#%% stop preview acquisiti|on
m.stop_preview()

m.block_laser()

#%% set these parameters for the current sample and objective to create filenames

sample = "sample"
region = "region"
indicator = "iGluSnFR3"
power = "15"
objective = "25x"
notes = "-"

basename = sample + "_" + region + "_" + indicator + "_" + objective + "_p" + power + "_" + notes

#%% acquire stack with more patterns than can be saved in the GLV (currently sending patterns from computer memory before every plane)
z1 = 0
n1 = m.z_steps_full-(2*z1)
m.syncRate = 50e3
m.sampleRate = 2000e3
nframes = 5

stackname = str(n1) + "planes_z1_" + str(z1) + "_" + str(m.def_min) + "to" + str(m.def_max) + "_syncRate_" + str(m.syncRate/1000) + "kHz_" + str(nframes) + "frames"
name = basename + "_" + stackname
m.is_stack = True
m.stack_save1 = m.acquire_overview_stack(name, nframes, n1, z1)

m.block_laser()

#%% make a montage of stack with >65k patterns for location selection
mn = m.make_montage_3(m.stack_save)

m.is_stack = True

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
app = QApplication([])
win = QMainWindow()

central_win = QWidget()
layout = QHBoxLayout()
central_win.setLayout(layout)
win.setCentralWidget(central_win)

ip = ImagePlot()
layout.addWidget(ip)  

levels = [0, 2e3]
ip.setImage(mn, levels)

print(ip.points) 
win.show()

#%% calculate targets from selected points
m.n_grid = 1
m.d_grid = 1 #in pixels

m.targets = m.get_targets_stack(ip.points, n_grid = m.n_grid, d_grid = m.d_grid)
m.n_targets = len(m.targets)

#%% acquire traces at target locations
m.syncRate = 340e3
m.sampleRate =1020e3

m.time_sec = 10
m.trace_rate = m.syncRate/m.n_targets
m.timepoints = int(m.time_sec*m.trace_rate)
new = True

notes = ""

tracename = "traces"+ str(m.syncRate/1000) + "kHz_" + str(m.time_sec) + "s_grid" + str(m.n_grid) + "_" + notes
name = basename + "_" + tracename

m.acquire_traces(m.targets, name, m.timepoints, new, corr = m.corr060723_1753_40x , plot = False)

m.block_laser()

#%%
m.bud.poll('POWERDOWN')

#%%
del m      