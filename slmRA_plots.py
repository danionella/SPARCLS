# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:16:28 2022

@author: jlab
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
import pyqtgraph as pg


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class ImagePlot(pg.GraphicsLayoutWidget):

    def __init__(self):
        super(ImagePlot, self).__init__()
        self.p1 = pg.PlotItem(title = "numupdates = 0")
        self.p1.setLabel(axis='left', text='x [points]')
        self.p1.setLabel(axis='bottom', text='y [points]')
        self.addItem(self.p1)
        self.p1.vb.invertY(True) # Images need inverted Y axis
        

        # Use ScatterPlotItem to draw points
        self.scatterItem = pg.ScatterPlotItem(
            size=5, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        self.scatterItem.setZValue(2) # Ensure scatterPlotItem is always at top
        self.points = [] # Record Points

        self.p1.addItem(self.scatterItem)


    def setImage(self, data, levels = [0, 1000]):
        
        self.p1.clear()

        self.image_item = pg.ImageItem(data[:,:])
        self.image_item.setLevels(levels)
        
        self.p1.addItem(self.image_item)
        self.p1.addItem(self.scatterItem)

    def mousePressEvent(self, event):

        point = self.p1.vb.mapSceneToView(event.pos()) # get the point clicked
        # Get pixel position of the mouse click
        x, y = int(point.x()), int(point.y())

        self.points.append([x, y])
        self.scatterItem.setData(pos=self.points)
        print('x = {}, y = {}'.format(x,y))
        super().mousePressEvent(event)

    

if __name__ == "__main__":

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication([])
    win = QMainWindow()

    central_win = QWidget()
    layout = QHBoxLayout()
    central_win.setLayout(layout)
    win.setCentralWidget(central_win)

    image_plot1 = ImagePlot()
    image_plot2 = ImagePlot()
    layout.addWidget(image_plot1)
    layout.addWidget(image_plot2)

    win.show()
    
    exit(app.exec_())
    QApplication.quit()

    if (sys.flags.interactive != 1) or not hasattr(Qt.QtCore, "PYQT_VERSION"):
        QApplication.instance().exec_()