# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:54:41 2022

@author: jlab
"""

import serial
import numpy as np
import time
import clr, sys
clr.AddReference(r'C:\Program Files (x86)\Cypress\EZ-USB FX3 SDK\1.3\bin\CyUSB')
import CyUSB


class GlvUSB:
    def __init__(self, dev_idx=0):
        device_list = CyUSB.USBDeviceList(CyUSB.CyConst.DEVICES_CYUSB.to_bytes(1, sys.byteorder))
        self.dev = device_list.get_Item(dev_idx)
        assert self.dev is not None
        self.dev.BulkOutEndPt.TimeOut = 1000
        self.msg_size = np.int32(4096)
        self.buf = bytearray(self.msg_size)
    
    def upload_pattern(self, buf):
        #assert len(pixvals) == 1088
        err = self.dev.BulkOutEndPt.XferData(buf, self.msg_size)
        return err
    
    
class HSBuddy:
    def __init__(self, comport, verbose=True):
        self.port = serial.Serial(comport, 115200, timeout=1)
        self.verbose = verbose
        self.glvusb = GlvUSB()
        
    def __del__(self):
        self.port.close()
        
    def write(self, cmd):
        self.port.flushInput()
        self.port.write(str.encode(cmd+'\r'))
        
    def read_answer(self):
        out = self.port.read_until(b'\r>')
        return out
    
    def poll(self, cmd):
        self.port.flushOutput()
        self.write(cmd)
        out = self.read_answer().decode("utf-8").splitlines()
        out = '\n'.join(out)
        if self.verbose: 
            print(out)
        else:
            return out
        
    def status(self):
        return self.poll('STAT') 
    
    def bootup(self):
        return self.poll('BOOTUP')
    
    def vddah(self):
        return self.poll('VDDAH 350')
    
    
    
    
class patterns:
    def calculate_pattern_from_position_def(wl,dx,dy,f,def_f,x0,y0):
        #y - first pass SLM, short axis
        #x - second pass SLM, long axis
        # dx, dy, f in um
        # wl wavelength in um
        # 0.01843 = lambda/pixel width
        thetax = np.arctan(dx/f);
        thetay = np.arctan(dy/f);
        defocusfactor = def_f;
        
        rdist_x = np.linspace((-272+x0)/272,(272+x0)/272,544)
        rdist_y = np.linspace((-272+y0)/272,(272+y0)/272,544)
        
        def_x = (defocusfactor*rdist_x**2)
        def_y = (defocusfactor*rdist_y**2)
        
        phi_x = np.empty(544)
        phi_y = np.empty(544)
        c = wl/25.5; #wavelength/pixel pitch in um
        
        if thetax == 0: 
            Nx = 0
            phi_x [:] = def_x#%(2*np.pi);
            if thetay == 0:
                phi_y [:] = def_y#%(2*np.pi);
                Ny = 0
            else:
                Ny = c/(np.sin(thetay))
                phi_y = (np.linspace(0,2*np.pi*544/Ny,544)+def_y)#%(2*np.pi)
                
        else:
            Nx = c/(np.sin(thetax));
            phi_x = (np.linspace(0,2*np.pi*544/Nx,544)+def_x)#%(2*np.pi)
            if thetay == 0:
                phi_y [:] = def_y#%(2*np.pi);
                Ny = 0
            else:
                Ny = c/(np.sin(thetay))
                phi_y = (np.linspace(0,2*np.pi*544/Ny,544)+def_y)#%(2*np.pi)
                                        
        phi = np.concatenate([phi_y, phi_x]) 
        return Ny,Nx,phi
    
    def calculate_pixvals(pattern, calib_f, calib_p, wl, maxpi): #f=690 p=4
        pattern = pattern%(2*np.pi)
        maxpi_wl = maxpi*(473/wl);
        pattern_adj = pattern+((maxpi_wl-2)*np.pi)/2
        pattern_adj[pattern_adj<0]= 0
        pattern_adj[pattern_adj>maxpi_wl*np.pi]= maxpi_wl*np.pi
        calib_fun = lambda x: calib_f*np.power(x,1/calib_p); 
        pixvals = calib_fun(pattern_adj*(wl/473));
        return pixvals
    
    
    
    
    
