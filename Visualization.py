import matplotlib
matplotlib.use('TkAgg') # This fixes a MAC-OS specific bug where figures get stuck under the IDE window.

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # !!!This is used even if it's grey!!!


#-----------------------------------------------------------------------------------------------------------------------
class Window3D():
    '''
    #
    '''

    def __init__(self, xLim, yLim, zLim, view = 'custom'):
        self.figureWindow = plt.figure()
        self.figureWindow.set_figwidth(9)
        self.figureWindow.set_figheight(10)
        self.axes = self.figureWindow.gca(projection = '3d')
        self.axes.set_aspect('equal')


        # Depending on the inputs adjusts the view-point to be from above the surface (2D) or to be from a custom
        # specific view angle.
        self.customView = [30, 30]
        if view == 'topdown':
            self.axes.view_init(90, 0)
        elif view == 'custom':
            self.axes.view_init(self.customView[0], self.customView[1])


        #
        # This code is needed in order for axis units to be equal.
        #
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([xLim[1] - xLim[0], yLim[1] - yLim[0], zLim[1] - zLim[0]]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xLim[1] + xLim[0])
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (yLim[1] + yLim[0])
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zLim[1] + zLim[0])
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.axes.plot([xb], [yb], [zb], 'w')

    def Delay(self, value):
        plt.pause(value)


    def Keep(self, status = True):
        '''
        Used if the plot window is to be left open after the program has finished.
        '''
        if status:
            plt.show()


#-----------------------------------------------------------------------------------------------------------------------
class Visualizer3D():
    '''
    The parent class of all custom made 3D visualization functions.
    '''
    def __init__(self, window):
        self.axes = window.axes
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')


    def Update(self):
        return NotImplementedError


#-----------------------------------------------------------------------------------------------------------------------
class Surf(Visualizer3D):
    '''
    #
    '''
    def __init__(self, window, x=None, y=None, z=None):
        super().__init__(window)
        if z is None:
            z = np.zeros([2, 2])
            if x is None:
                x = np.zeros([2, 2])
            if y is None:
                y = np.zeros([2, 2])
            self.visible = False
        else:
            x = np.arange(0, z.shape[0], 1)
            y = np.arange(0, z.shape[1], 1)
            x, y = np.meshgrid(x, y)
            self.visible = True


        self.surface = self.axes.plot_surface(x, y, z, cmap = cm.gray, linewidth = 0, antialiased = False)
        # rcount=500, ccount=500 # use these parameters for the plot_surface() for finer resolution.
        if self.visible:
            plt.pause(0.0000001)


    def Update(self, z):
        '''
        When called the function updates the z-values for the surface object and visualizes that change.
        '''
        tmpVal = z.shape[0]
        x = np.arange(0, z.shape[0], 1)
        y = np.arange(0, z.shape[1], 1)
        x, y = np.meshgrid(x, y)

        # The old plot_surface object is removed and another is created. Instead of deleting the old object and
        # creating a new object the object should just be changed, not sure how to do this in python.
        self.surface.remove()
        self.surface = self.axes.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False, rcount = 500, ccount = 500)
        plt.pause(0.0000001)


#-----------------------------------------------------------------------------------------------------------------------
class Lines(Visualizer3D):
    '''
    #
    '''
    def __init__(self, window, numberOfLines):
        super().__init__(window)
        numberOfLines = 1
        x = [0.0]
        y = [0.0]
        z = [0.0]
        #self.lines = [self.axes.plot(x, y, z, marker='.', markerSize=0.8, linewidth=0.4, color='red') for index in range(numberOfLines)]
        self.lines = self.axes.plot(x, y, z, marker='.', markerSize=0.2, linewidth=0.4, color='red')


    def Update(self, data):
        '''
        data is a list of arrays. Each array represent a specific line.
        :param data:
        :return:
        '''
        self.lines[0].set_data(data[0:2, ::1])
        self.lines[0].set_3d_properties(data[2, ::1])


#-----------------------------------------------------------------------------------------------------------------------
def PrepareLines(drops):
    '''
    The function appends the trail data from each drops into a single long trail. Between each short-trail a column of
    None is added, this is done in order to separate the data once it is visualized. At locations where a trail circles
    the map using the periodic boundary conditions a None is also added in order to make the visualization look better.

    :param drops:
    :return:
    '''

    trailData = np.zeros((3, 1))
    trailData[0:3, -1] = None
    # The trails are appended together, separated by Nones.
    for drop in drops:
        #trailData = np.append(trailData, drop.trailData[0:3, :iStep + 1], axis=1) # used when animating the trail, step by step.
        trailData = np.append(trailData, drop.trailData, axis=1)
        trailData[0:3, -1] = None  # Separates each line.
    trailDataCopy = trailData


    # Nones are inserted to prevent lines from wrapping the entire map.
    for iElement in range(trailData.shape[1] - 1):
        if trailDataCopy[0, iElement] is not None:
            if np.sqrt((trailDataCopy[0, iElement] - trailDataCopy[0, iElement + 1]) ** 2
                               + (trailDataCopy[1, iElement] - trailDataCopy[1, iElement + 1]) ** 2) > 1.5:
                trailData[0:3, iElement] = None
    return trailData







#-----------------------------------------------------------------------------------------------------------------------
class SurfaceVisualizer():
    '''
    The class is used when surfaces are to be used. The class contains one surface object. The surface can be updated
    using the Update() function. In order for the aspect ratio to work properly an invisible box is created around the
    surface, this box ensures that the aspect ratio is correct. The use of this box disables the use of custom x-, y-
    and z-limits.
    #
    # A function to change the custom view-point value should be added.
    # A function to save the image to file should be added.
    # Support for both custom view-point and topdown should be added (2 axes).
    #
    '''


    def __init__(self, xLim, yLim, zLim, view = 'custom'):
        self.figureWindow = plt.figure()
        self.axes = self.figureWindow.gca(projection = '3d')
        self.axes.set_aspect('equal')


        # Depending on the inputs adjusts the view-point to be from above the surface (2D) or to be from a custom
        # specific view angle.
        self.customView = [30, 30]
        if view == 'topdown':
            self.axes.view_init(90, 0)
        elif view == 'custom':
            self.axes.view_init(self.customView[0], self.customView[1])


        x = np.zeros([2, 2])
        y = np.zeros([2, 2])
        z = np.zeros([2, 2])
        self.surface = self.axes.plot_surface(x, y, z, cmap = cm.gray, linewidth = 0, antialiased = False)


        #
        # This code is needed in order for axis units to be equal.
        #
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([xLim[1] - xLim[0], yLim[1] - yLim[0], zLim[1] - zLim[0]]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (xLim[1] + xLim[0])
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (yLim[1] + yLim[0])
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (zLim[1] + zLim[0])
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.axes.plot([xb], [yb], [zb], 'w')


    def Update(self, z):
        '''
        When called the function updates the z-values for the surface object and visualizes that change.
        '''
        tmpVal = z.shape[0]
        x = np.arange(0, tmpVal, 1)
        y = np.arange(0, tmpVal, 1)
        x, y = np.meshgrid(x, y)

        # The old plot_surface object is removed and another is created. Instead of deleting the old object and
        # creating a new object the object should just be changed, not sure how to do this in python.
        self.surface.remove()
        self.surface = self.axes.plot_surface(x, y, z, cmap=cm.gray, linewidth=0, antialiased=False)
        plt.pause(0.0000001)




