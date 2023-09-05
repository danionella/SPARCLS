# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:24:11 2022

@author: jlab
"""
import numpy as np
import pyqtgraph as pg
import json
import logging
import os
import time
import arrow
from slmRA_hardware import HSBuddy, patterns #GlvUSB
from slmRA_plots import ImagePlot
from nidaqmx import Task, stream_readers, stream_writers
from nidaqmx.constants import AcquisitionType, Edge, FrequencyUnits, Level
import tkinter as tk
from tkinter import filedialog
import tifffile
from skimage.util import montage

from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0,"C:/Users/jlab/.spyder-py3/RandomAccess/montages")
import montages as mn

import System

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout


logger = logging.getLogger(__name__)


class SLM_RA(object):
    '''
    Control class for SLMRA microscope
    '''
    
    logger.debug("Loading config file")
    
    def __init__(self, conf_p, save_p, hardware):
        #load config file
        try:
            with open(os.path.join(conf_p, "slmRA_config.json")) as f:
                try:
                    self.conf = json.load(f)
                except OSError:
                    logger.error("Error in Json file")
        except OSError: #or FileNotFoundError
                logger.error("Config File Not Found")
        
        self.hardware = hardware
        
        #set config parameters
        self.savePath = save_p
        self.dev = self.conf["DAQDevice"]
        # self.sampleRate = self.conf["Timing"]["DAQRate"]
        self.syncOut = self.conf["Channels"]["SyncOut"]
        self.GLVtrig = self.conf["Channels"]["GLVtrig"]
        self.PMT = self.conf["Channels"]["PMT"]
        # self.syncRate = self.conf["Timing"]["SyncRate"]
        self.dutyCycle = self.conf["Timing"]["DutyCycle"]
        self.bufferSize = self.conf["Timing"]["BufferSize"]

        self.wl = self.conf["Calib"]["Wavelength"]
        self.f_cyl = self.conf["Calib"]["CylLens"]
        self.calib_f = self.conf["Calib"]["Factor"]
        self.calib_p = self.conf["Calib"]["Power"]
        self.maxPi = self.conf["Calib"]["MaxPi"]
        self.points = 10000
        
        self.msg_size = np.int32(4096)
        self.buf = bytearray(self.msg_size)
        
        self.path_40x = None
        self.path_25x = None
        
        self.wl = 940
        self.f_cyl = 32e3
        self.calib_f = 690
        self.calib_p = 4
        self.maxpi = 1.535
        
        #initialise GLV
        if self.hardware:
            self.bud = HSBuddy('COM1')
            self.bud.poll('STAT')
            self.bud.poll('COLSOURCE EXT CNT EXT')
            self.bud.poll('USB 0 65000 1') # USB <dir> <start> <count> tells Cosmo board to expect data via USB, dir: 0 for sending, 1 for receiving, start: start column, count: number of columns
            print(self.bud.read_answer())
            
            Nx,Ny,phi = patterns.calculate_pattern_from_position_def(self.wl/1000, 10, 10, self.f_cyl, 0, 0, -120)
            pixvals = patterns.calculate_pixvals(phi,self.calib_f, self.calib_p, self.wl, self.maxpi)
            pixvals = pixvals.astype('>i2')
            
            self.targetstack = np.zeros((1, len(self.buf)), dtype='int8')
            self.buf[:1088*2] = pixvals.reshape(2,272,2).transpose(1,0,2).flatten().tobytes()
            self.targetstack[0,:] = self.buf
        
            self.buf_cs2 = System.Array.CreateInstance(System.Byte, self.targetstack.size)
    
            ptr_np = System.IntPtr.__overloads__[System.Int64](self.targetstack.__array_interface__['data'][0])
            System.Runtime.InteropServices.Marshal.Copy(ptr_np, self.buf_cs2, 0, self.targetstack.size)
            
            self.bud.glvusb.dev.BulkOutEndPt.XferData(self.buf_cs2, np.int32(self.targetstack.size))
            

        
    def setup_tasks(self):
        self.task_shutter = Task()
        self.task_shutter.do_channels.add_do_chan("/" + self.conf["DAQDevice"] + "/" + self.conf["Channels"]["Shutter"])
        self.ShutterWriter=stream_writers.DigitalSingleChannelWriter(self.task_shutter.out_stream,auto_start=True)
        self.close_shutter = lambda: self.ShutterWriter.write_one_sample_one_line(False)
        self.open_shutter = lambda: self.ShutterWriter.write_one_sample_one_line(True)
        self.close_shutter()
        
        
        self.task_sync = Task()
        self.sync_channel = self.task_sync.co_channels.add_co_pulse_chan_freq(
            "{}/{}".format(self.dev, self.syncOut), units=FrequencyUnits.HZ, idle_state=Level.LOW, initial_delay=0.0, freq=self.syncRate, duty_cycle=self.dutyCycle)
        self.sync_channel.co_pulse_term = "/Dev2/PFI4"
        self.task_sync.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=int(self.bufferSize))
        self.task_sync.triggers.start_trigger.cfg_dig_edge_start_trig("/{}/ai/StartTrigger".format(self.dev), Edge.RISING)
        
        self.task_in = Task()
        self.task_in.ai_channels.add_ai_voltage_chan("{}/{}".format(self.dev, self.PMT))
        self.task_in.timing.cfg_samp_clk_timing(self.sampleRate,samps_per_chan=self.points*2, sample_mode=AcquisitionType.CONTINUOUS)
        
        
    def stop_tasks(self):
        self.close_shutter()
        try:
            self.task_sync.stop()
        except:
            pass
        
        try:
            self.task_in.stop()
        except:
            pass
        
        try:
            self.task_shutter.stop()
        except:
            pass
        
              
    def close_tasks(self):
        try:
            self.task_sync.close()
        except:
            logger.warning("Task Already Closed")
            pass
        
        try:
            self.task_in.close()
        except:
            logger.warning("Task Already Closed")
            pass
        
        try:
            self.task_shutter.close()
        except:
            logger.warning("Task Already Closed")
            pass
        
        
    def calculate_stacks(self, corr = None):
        self.z_steps = self.z_steps_full
        self.points = self.x_steps*self.y_steps*self.z_steps
        
        self.xs = np.linspace(self.x_min,self.x_max, self.x_steps)
        self.ys = np.linspace(self.y_min, self.y_max, self.y_steps)
        self.zs_full = np.linspace(self.def_min,self.def_max,self.z_steps)
        
        self.stack = np.zeros((len(self.zs_full), len(self.ys), len(self.xs), len(self.buf)), dtype='int8')
    
        
        count = 0
        t0 = time.perf_counter()
        pz = 0
        
        for def_f in self.zs_full:
            py = 0
            for dy in self.ys: # max 921 (N=2), (0,600,61)
                px = 0
                for dx in self.xs: # max 921 (N=2), (-40,-280,25)
                    Nx,Ny,phi = patterns.calculate_pattern_from_position_def(self.wl/1000, dx, dy, self.f_cyl, def_f, self.x_0, self.y_0)
                    if corr is not None:
                        phi = (phi + corr)#%(2*np.pi)
                    pixvals = patterns.calculate_pixvals(phi,self.calib_f, self.calib_p, self.wl, self.maxpi)
                    count = count+1;
                    pixvals = pixvals.astype('>i2')
                    self.buf[:1088*2] = pixvals.reshape(2,272,2).transpose(1,0,2).flatten().tobytes()
                    self.stack[pz, py, px, :] =self.buf
                    px += 1
                py += 1
            pz += 1
        t3 = time.perf_counter()
        t = t3-t0 
        
        return count, t
        
    
    def send_patterns_n(self, corr = None, z = 'all', n = 0): # to do: group (calib, x, y, def, e.g. calib.f)
        #send patterns to Cosmo boards
        
        if z == 'all':
            self.zs = self.zs_full
            self.send_stack = self.stack
        else:
            self.zs = self.zs_full[(z):(z+n)]
            self.points = self.x_steps*self.y_steps*len(self.zs)
            self.z_steps = len(self.zs)
            self.send_stack = self.stack[(z):(z+n), :, :]
        if self.points > 65536:
            print("Too many points")
            pass
        
        if self.hardware:
            self.bud.poll('USB 0 0 {}'.format(self.points)) # USB <dir> <start> <count> tells Cosmo board to expect data via USB, dir: 0 for sending, 1 for receiving, start: start column, count: number of columns
            print(self.bud.read_answer()) # receiving 
        
        #send patterns for 2D scan (x = short axis, vertical)
        n_pack = 1000
        count = 0
        msgs = np.arange(0,self.points,n_pack)
        N = len(msgs)
        msgs = np.append(msgs,self.points)
        t0 = time.perf_counter()
        
        self.fl_stack = self.send_stack.flatten()
        self.substack = self.fl_stack[4096*msgs[n]:4096*msgs[n+1]-1]
        #self.buf_cs = System.Array.CreateInstance(System.Byte, self.stack.size)
        self.buf_cs1 = System.Array.CreateInstance(System.Byte, self.substack.size)
        
        for n in range(N):
            self.substack = self.fl_stack[4096*msgs[n]:4096*msgs[n+1]-1]
            self.buf_cs1 = System.Array.CreateInstance(System.Byte, self.substack.size)
            ptr_np = System.IntPtr.__overloads__[System.Int64](self.substack.__array_interface__['data'][0])
            System.Runtime.InteropServices.Marshal.Copy(ptr_np, self.buf_cs1, 0, self.substack.size)
            self.bud.glvusb.dev.BulkOutEndPt.XferData(self.buf_cs1, np.int32(self.substack.size))
        
        t3 = time.perf_counter()
        tg = t3-t0
        return count, tg
    
    
    def send_targets(self, targets, corr = None): # to do: group (calib, x, y, def, e.g. calib.f)
        #send patterns to Cosmo boards
        ntargets = np.shape(targets)[0]

        if self.hardware:
            self.bud.poll('USB 0 0 {}'.format(ntargets)) # USB <dir> <start> <count> tells Cosmo board to expect data via USB, dir: 0 for sending, 1 for receiving, start: start column, count: number of columns
            print(self.bud.read_answer()) # receiving 
        
        #send patterns for 2D scan (x = short axis, vertical)
        t1 = time.perf_counter()
        self.targetstack = np.zeros((ntargets, len(self.buf)), dtype='int8')
        
        for n in range(ntargets):
            def_f = targets[n,2]
            dy = targets[n,1]
            dx = targets[n,0]
            Nx,Ny,phi = patterns.calculate_pattern_from_position_def(self.wl/1000, dx, dy, self.f_cyl, def_f, self.x_0, self.y_0)
            if corr is not None:
                phi = (phi + corr)%(2*np.pi)
            pixvals = patterns.calculate_pixvals(phi,self.calib_f, self.calib_p, self.wl, self.maxpi)
            pixvals = pixvals.astype('>i2')
            self.buf[:1088*2] = pixvals.reshape(2,272,2).transpose(1,0,2).flatten().tobytes()
            self.targetstack[n,:] = self.buf
                
        t2 = time.perf_counter()
        t = t2-t1
        
        self.buf_cs2 = System.Array.CreateInstance(System.Byte, self.targetstack.size)
        ptr_np = System.IntPtr.__overloads__[System.Int64](self.targetstack.__array_interface__['data'][0])
        System.Runtime.InteropServices.Marshal.Copy(ptr_np, self.buf_cs2, 0, self.targetstack.size)
        
        if self.hardware:
            self.bud.glvusb.dev.BulkOutEndPt.XferData(self.buf_cs2, np.int32(self.targetstack.size))
        return t
    
    
    def send_targets_Zernike(self, targets, mode, maxf, corr): # to do: group (calib, x, y, def, e.g. calib.f)
        #send patterns to Cosmo boards
        self.ntargets = np.shape(targets)[0]
        
        npoints = self.ntargets*self.nphasesteps

        if self.hardware:
            self.bud.poll('USB 0 0 {}'.format(npoints)) # USB <dir> <start> <count> tells Cosmo board to expect data via USB, dir: 0 for sending, 1 for receiving, start: start column, count: number of columns
            print(self.bud.read_answer()) # receiving 
            
        self.targetstack = np.zeros((npoints, len(self.buf)), dtype='int8')
        
        #send patterns for 2D scan (x = short axis, vertical)
        t1 = time.perf_counter()
        
        for n in range(self.ntargets):
            def_f = targets[n,2]
            dy = targets[n,1]
            dx = targets[n,0]
            Nx,Ny,phi = patterns.calculate_pattern_from_position_def(self.wl/1000, dx, dy, self.f_cyl, def_f, self.x_0, self.y_0)
            k = 0
            for factor in self.factors:
                Z = self.calculate_Zernike(mode, factor)
                phiZ = (phi + Z)#%(2*np.pi)
                if corr is not None:
                    phiZ = (phiZ + corr)#%(2*np.pi)
                pixvals = patterns.calculate_pixvals(phiZ,self.calib_f, self.calib_p, self.wl, self.maxpi)
                pixvals = pixvals.astype('>i2')
                self.buf[:1088*2] = pixvals.reshape(2,272,2).transpose(1,0,2).flatten().tobytes()
                self.targetstack[n*self.nphasesteps+k,:] = self.buf
                k += 1
                     
        t2 = time.perf_counter()
        t = t2-t1
        
        self.buf_cs2 = System.Array.CreateInstance(System.Byte, self.targetstack.size)
        ptr_np = System.IntPtr.__overloads__[System.Int64](self.targetstack.__array_interface__['data'][0])
        System.Runtime.InteropServices.Marshal.Copy(ptr_np, self.buf_cs2, 0, self.targetstack.size)
        
        if self.hardware:
            self.bud.glvusb.dev.BulkOutEndPt.XferData(self.buf_cs2, np.int32(self.targetstack.size))
            
        return t
        
    def calculate_Zernike(self, mode, factor):
        #X = np.linspace(-1,1,544)
        #Y = np.linspace(-1,1,544)
       
        X = np.linspace((-272+self.x_0)/272,(272+self.x_0)/272,544)
        Y = np.linspace((-272+self.y_0)/272,(272+self.y_0)/272,544)
        
        f = factor
        
        if mode == 1: #N1, tilt
            Zx = 0*X
            Zy = 2*Y
        elif mode == 2: #N2,tip
            Zx = 2*X
            Zy = 0*Y
        elif mode == 3: #N4, defocus
            Zx = np.sqrt(3)*2*X**2-0.5
            Zy = np.sqrt(3)*2*Y**2-0.5
        elif mode == 4: #N5 vertical astigmatism
            Zx = np.sqrt(6)*X**2
            Zy = -1*np.sqrt(6)*Y**2
        elif mode == 5:#N6 vertical trefoil
            Zx = 0*X
            Zy = -1*np.sqrt(8)*Y**3
        elif mode == 6:#N7 vertical coma
            Zx = 0*X
            Zy = (3*Y**3-2*Y)
        elif mode == 7:#N8 horizontal coma
            Zy = 0*Y
            Zx = (3*X**3-2*X)
        elif mode == 8:#N9 oblique trefoil
            Zx = X**3
            Zy = 0*Y
        elif mode == 9:#N12 1st spherical abb.
            Zx = np.sqrt(5)*(6*X**4-6*X**2+1)
            Zy = np.sqrt(5)*(6*Y**4+1)
        elif mode == 10:#N13 vertical 2nd astigmat.
            Zx = np.sqrt(10)*(4*X**4-3*X**2)
            Zy = np.sqrt(10)*(-4*Y**4+3*Y**2)
        elif mode == 11:#N14 vertical quadrafoil
            Zx = np.sqrt(10)*X**4
            Zy = np.sqrt(10)*Y**4
        else:
            print("not defined")
      
        Z = f*np.concatenate([Zy,Zx])#%(2*np.pi)

        return Z 
    
    def calculate_correction(self, modes, factors):
        corr1 = np.zeros(1088)
        for mod in modes:
            corr1 += self.calculate_Zernike(mod,factors[mod-modes[0]])
        corr = corr1
        return corr
        
    def preview(self, scale, new):
        
        if self.points > 131072:
            print("Too many points")
            pass
        self.scale = scale
        self.setup_tasks()
        if new == True:
            self.bud.send_patterns()
            time.sleep(self.points*0.0012)
        # self.bud.poll('COLSOURCE EXT CNT EXT')
        # time.sleep(2)
        self.bud.poll('LOOPLUT 0 {} 0'.format(self.points-1)) 
        
        self.binsize = round(self.sampleRate/self.syncRate)
        
        self.data_in = np.zeros(self.points*self.binsize, dtype = np.float64)
        
        self.plotI = pg.PlotItem(title = "numupdates = 0")
        self.plotI.setLabel(axis='left', text='x [points]')
        self.plotI.setLabel(axis='bottom', text='y [points]')
        self.plot = pg.ImageView(view = self.plotI)
        self.plot.show()
        
        self.reader = stream_readers.AnalogSingleChannelReader(self.task_in.in_stream)
        self.prevMA = np.zeros((self.z_steps,self.y_steps,self.x_steps))
        self.numUpdates=1
        self.timer = QtCore.QTimer()
        self.timer.setInterval((self.points*self.binsize*100)/(self.sampleRate))
        
        self.timer.timeout.connect(self.update_plot)
        
        self.open_shutter()
        self.task_sync.start()
        self.task_in.start()
   
        print('Opening task (ctrl-c to stop)')
        
        self.update_plot()
        self.prevMA = self.data

        self.timer.start()

        
    def stop_preview(self):
        self.timer.stop()
        self.plotI.close()
        self.bud.poll('/')
        self.close_shutter()
        self.stop_tasks()
        self.close_tasks()
        
    def update_plot(self, a = 0.5):
        self.reader.read_many_sample(self.data_in, self.points*self.binsize)
        data = (-1e4)*self.data_in
        data = np.reshape(data, (-1, self.binsize)).mean(axis = 1) #binning
        self.meanF= np.mean(data)
        self.data= np.reshape(data,(self.z_steps,self.y_steps,self.x_steps)) 
        pos = self.plot.currentIndex
        if self.gauss == True:
            self.data = gaussian_filter(self.data, sigma = 1)
            
        self.MA = a*self.data + (1-a)*self.prevMA #moving average
        if self.rollAv == True:
            self.plot.setImage(self.MA, autoLevels=False, levels=(0,self.scale)) #levels = (0,2)
        else:
            self.plot.setImage(self.data, autoLevels=False, levels=(0,self.scale))
        self.plot.setCurrentIndex(pos)
        self.numUpdates = self.numUpdates+1
        #self.plotI.setTitle(title = "numupdates = {}".format(self.numUpdates))
        self.plotI.setTitle(title = "mean = {}".format(self.meanF))
        self.prevMA = self.MA
        
      
    def load_data(self, file_path = None):
        if file_path == None:
            root = tk.Tk()
            #root.withdraw()
            file_path = filedialog.askopenfilename()
            root.destroy()
        data_l = np.load(file_path)
        return data_l, file_path
    
    def load_stack(self):
        root = tk.Tk()
        #root.withdraw()
        filepaths = filedialog.askopenfilenames()
        root.destroy()
        for filepath in filepaths[:1]:
            data1 = np.load(filepath)
            size= data1.shape
        nfiles = len(filepaths)
        data = np.zeros((nfiles, size[0],size[1], size[2], size[3]))
        
        n = 0
        for filepath in filepaths:
            data[n,:,:,:,:] = np.load(filepath)
            n +=1
        return data
        
    def make_montage(self, data, format = 'square'):
        if len(data.shape) == 4:
            mon = montage((np.mean(data[:,:,:,:], axis = 0)))
            if format == 'single':
                mon = montage((np.mean(data[:,:,:,:], axis = 0)), grid_shape = (1,data.shape[0]))
        elif len(data.shape) == 3:
            mon = montage(data, fill = 2550, padding_width = 0)
            if format == 'single':
                mon =  montage(data, grid_shape = (1,data.shape[0]))
        else:
            print("wrong format")
        #plt.imshow(mn.T)
        return mon
    
    def make_montage_3(self, data):
        mon = montage(data)
        return mon
    
    def get_points(self, montage):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
         
        #data = np.random.rand(x_steps,y_steps, 10)   
        #plot.setImage(data.T)
        
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        app = QApplication([])
        win = QMainWindow()
        
        central_win = QWidget()
        layout = QHBoxLayout()
        central_win.setLayout(layout)
        win.setCentralWidget(central_win)
        
        ip = ImagePlot()
        layout.addWidget(ip)
        ip.setImage(mn)
        
        
        print(ip.points)
        win.show()
        #exit(app.exec_())
        #QApplication.quit()
        return ip.points
    
    
    def get_targets_stack(self, points, n_grid, d_grid):
        #n_grid = 1: single point, n_grid = 5, 2D 5x5 grid
        #d_grid: distance between grid points, in um
        self.n_grid = n_grid
        self.d_grid = d_grid
        self.point_targets = points
        grid_mn = np.ceil(np.sqrt(len(self.z_full)))
        np.shape(self.point_targets)[0]
        arr = np.zeros((np.shape(self.point_targets)[0]*n_grid**2, 3), dtype=np.int16)
        n = 0
        
        if n_grid ==1:
            fpoints= points
            
        elif n_grid > 1:
            fpoints = []
            N = (n_grid-1)/2
            
            xgr = np.linspace(-d_grid*N,d_grid*N,n_grid)
            ygr =  np.linspace(-d_grid*N,d_grid*N,n_grid)

            p1 = 0
            for p in points:
                for nx1 in range(n_grid):
                    for ny1 in range(n_grid):
                        pt = [p[0]+xgr[nx1],p[1]+ygr[ny1]]
                        fpoints.append(pt)
                p1 +=1
            
                
        for p in fpoints:
            dx = int(p[0]%self.y_steps)
            dy = int(p[1]%self.x_steps)
            dz = int(p[1]//self.x_steps + grid_mn*(p[0]//self.y_steps))
            arr[n] = [dx,dy,dz]
            print(arr[n])
            n = n+1

        targets = np.zeros(np.shape(arr))
        n = 0
        for (nx, ny, nz) in arr:
            targets[n] = (self.xs[ny],self.ys[nx],self.zs_stack[nz])
            print(targets[n])
            n = n+1
                
        return targets
    


    def acquire_traces(self, targets, name, timepoints, new, corr = None, AO=1, plot = True, notes = None, todisk = True):
        # plot overview stack (still image) with numbered points
        # plot numbered or color coded traces (building up) next to it (if plot = True)
        
        dstr = arrow.now().format("YYYYMMDD_HHmm_")
        dirname = dstr + name

        self.ntargets = np.shape(targets)[0]
        
        parDict = {
            "filename": name,
            "xShift": self.x_0,
            "yShift": self.y_0,
            "xMin": self.x_min,
            "xMax": self.x_max,
            "xSteps": self.x_steps,
            "yMin": self.y_min,
            "yMax": self.y_max,
            "ySteps": self.y_steps,
            "defMin": self.def_min,
            "defMax": self.def_max,
            "zSteps": self.z_steps,
            "newParams": new,
            "timepoints": timepoints,
            "sampleRate": self.sampleRate,
            "syncRate": self.syncRate,
            "wl": self.wl,
            "fCyl": self.f_cyl,
            "calibF": self.calib_f,
            "calibP": self.calib_p,
            "maxPi": self.maxPi,
            "n_targets": self.ntargets,
            "trace_time": self.time_sec,
            "trace_rate": self.trace_rate,
            "trace_timepoints": self.timepoints,
            "d_grid": self.d_grid,
            "n_grid": self.n_grid,
            "correction_path_40x": self.path_40x,
            "correction_path_25x": self.path_25x
        }
         
        if notes is not None:
            parDict["notes"] = notes
       
        parDict = {**parDict, **self.conf}
       
        self.setup_tasks()
        
        if new:
            self.send_targets(targets, corr)
            time.sleep(self.ntargets*0.0012)
            
        self.bud.poll('LOOPLUT 0 {} 0'.format((self.ntargets)-1))
            
        binstep = 100
            
        self.binsize = round(self.sampleRate/self.syncRate)
        self.data_in = np.zeros(self.ntargets*self.binsize*binstep, dtype = np.float64) 

        self.reader = stream_readers.AnalogSingleChannelReader(self.task_in.in_stream)

        self.traces_save = np.ndarray((timepoints, self.ntargets), dtype = np.uint16)
        
        self.open_shutter()
        time.sleep(1)
        self.task_sync.start()
        self.task_in.start()

        print('Opening task (ctrl-c to stop)')
        
        
        steps = int(timepoints/binstep)
        
        for n1 in range(steps):
            self.reader.read_many_sample(self.data_in, self.ntargets*self.binsize*binstep)
            data = (-1e4)*self.data_in
            data = np.reshape(data, (-1, self.binsize)).mean(axis = 1) #binning
            #self.data= np.reshape(data,(self.z_steps,self.y_steps,self.x_steps))
            for ms in range(binstep):
                n2 = n1*binstep+ms
                self.traces_save[n2,:] = data[ms*self.ntargets:(ms+1)*self.ntargets].astype(np.uint16)
            
        self.bud.poll('/')
        self.close_shutter()
        self.stop_tasks()
        self.close_tasks()    
        
        if todisk:
            if not os.path.exists(os.path.join(self.savePath, dirname)):
                os.mkdir(os.path.join(self.savePath, dirname))
            with open(os.path.join(self.savePath, dirname, "meta_traces.json"), "w+") as f:
                json.dump(parDict, f, indent=4)
 
            self.array_p = os.path.join(self.savePath, dirname, "traces_np.npy")
            self.target_p = os.path.join(self.savePath, dirname, "targets_np.npy")
            self.point_p = os.path.join(self.savePath, dirname, "points_np.npy")
            np.save(self.array_p, self.traces_save)
            np.save(self.target_p, targets)
            np.save(self.point_p, self.point_targets)
                      
        else:
            pass

            
    
    def acquire_overview_stack(self, name, nframes, n1,z1, dstr = None, new = False, notes=None, todisk=True, z = "b0"): 
        """
        acquire overview scan
        """
        if dstr == None:
            dstr = arrow.now().format("YYYYMMDD_HHmm_")
        
        dirname = dstr + name
        
        self.frameRate = self.syncRate/self.points
        print(self.frameRate)
        
        parDict = {
            "filename": name,
            "xShift": self.x_0,
            "yShift": self.y_0,
            "xMin": self.x_min,
            "xMax": self.x_max,
            "xSteps": self.x_steps,
            "yMin": self.y_min,
            "yMax": self.y_max,
            "ySteps": self.y_steps,
            "defMin": self.def_min,
            "defMax": self.def_max,
            "zSteps": self.z_steps,
            "newParams": new,
            "nFrames": nframes,
            "sampleRate": self.sampleRate,
            "syncRate": self.syncRate,
            "wl": self.wl,
            "fCyl": self.f_cyl,
            "calibF": self.calib_f,
            "calibP": self.calib_p,
            "maxPi": self.maxPi,
            "frameRate": self.frameRate,
            "correction_path_40x": self.path_40x,
            "correction_path_15x": self.path_25x,
            "n1": n1,
            "startplane": z1
        }
         
        if notes is not None:
            parDict["notes"] = notes
       
        parDict = {**parDict, **self.conf}
        
        self.z_inc = np.arange(z1,n1+z1, 1).astype("int")
        self.z_full = np.arange(0,self.z_steps_full,1).astype("int")
        self.zs_stack = self.zs_full[self.z_full]
        self.stack_save = np.ndarray((len(self.zs_stack), self.y_steps, self.x_steps), dtype = np.uint16)
        self.stack_save_std = np.ndarray((len(self.zs_stack), self.y_steps, self.x_steps), dtype = np.uint16)
        self.stack_save_max = np.ndarray((len(self.zs_stack), self.y_steps, self.x_steps), dtype = np.uint16)
        for zs in self.z_inc:
            print(zs)
            self.send_patterns_n(z = zs, n = 1)
            frame = self.acquire_overview(name, nframes, dstr, new = False, z = str(zs))
            self.stack_save_std[zs,:,:] = np.squeeze(np.std(frame, axis = 0))
            self.stack_save_max[zs,:,:] = np.squeeze(np.max(frame, axis = 0))
            self.stack_save[zs,:,:] = np.squeeze(np.mean(frame, axis = 0))

       
        if todisk:
            if not os.path.exists(os.path.join(self.savePath, dirname)):
                os.mkdir(os.path.join(self.savePath, dirname))
            with open(os.path.join(self.savePath, dirname, "meta.json"), "w+") as f:
                json.dump(parDict, f, indent=4)

            self.corr_p = os.path.join(self.savePath, dirname, "corr_np.npy")
            self.stack_p =  os.path.join(self.savePath, dirname, "stack_np.npy")
            self.tiff_stack =  os.path.join(self.savePath, dirname, dstr+"stack.ome.tif")
            
            np.save(self.stack_p, self.stack_save)
            tifffile.imwrite(self.tiff_stack, self.stack_save.transpose(0,2,1), imagej = True, metadata={'fps': 1, 'axes': 'ZYX'}) #fps is placeholder
            np.save(self.corr_p, self.corr_n)
               
        else:
            pass
        
        return self.data_save
        
    def block_laser(self):
        self.bud.poll('COLSOURCE INT CNT EXT')
        self.bud.poll('GOLUT 65000 65000')
        self.bud.poll('COLSOURCE EXT CNT EXT')
        ans = self.bud.read_answer()
        return ans
        
        
    def stop(self):
        self.close_shutter()
        self.bud.poll("/")
        #stop tasks
        self.stop_tasks()

        
    def __del__(self):
        #self.bud.poll('POWERDOWN')
        self.stop_tasks()
        self.close_tasks()
        time.sleep(1)
        self.bud.port.close()
        del self.bud
        
        
        
