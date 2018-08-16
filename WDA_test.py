#
#From this sqript file the waterdrop algorithm can be tested.
#


import Simulation.Erosion.Hydrolic.WDA as WDA
import Simulation.Noise as Noise
import Visualization
import numpy as np
import time # Used to time code, like tic-toc.
import cProfile


mapSize = 512
initialMaximumHeight = 100
numberOfRuns = 1
numberOfDrops = 100
numberOfSteps = 64

displaySurface = False
displayTrail = False
performProfiling = False


maximumErosionRadius = 10  # This determines how many erosion templates should be created.


# The height map is generated from "simple noise".
heightMap = Noise.SimpleNoise(mapSize,3,2)
heightMap *= initialMaximumHeight
print('Noise has been generated')


# Create visualization objects
if displaySurface:
    initialWindow = Visualization.Window3D(xLim = [0, mapSize], yLim = [0, mapSize], zLim = [0, initialMaximumHeight], view ='topdown')
    mainWindow = Visualization.Window3D(xLim = [0, mapSize], yLim = [0, mapSize], zLim = [0, initialMaximumHeight], view ='topdown')
    initialMapSurface = Visualization.Surf(initialWindow, z = heightMap)
    mapSurface = Visualization.Surf(mainWindow, z = heightMap)
    if displayTrail:
        trailLines = Visualization.Lines(mainWindow, numberOfDrops)


# Creates templates used by all the drops.
WDA.WaterDrop.InitializeTemplates(maximumErosionRadius)
WDA.WaterDrop.LinkToHeightMap(heightMap)


if performProfiling:
    pr = cProfile.Profile()
    pr.enable()

tic = time.time()
for iRun in range(numberOfRuns):
    # Create the drops
    drops = [WDA.WaterDrop(mapSize, numberOfSteps=numberOfSteps, storeTrail=displayTrail, inertia=0.4, capacityMultiplier=30,
                           erosionRate=0.8, erosionRadius=2) for index in range(numberOfDrops)]
    # Move and animate the drops
    for iStep in range(numberOfSteps):
        for drop in drops:
            drop()
toc = time.time()
print('elapsed time : %s sec' % (toc - tic))

if performProfiling:
    pr.disable()
    pr.print_stats(2)


if displaySurface:
    mapSurface.Update(heightMap)


# The trails of the drops are visualized.
if displayTrail:
    trailData = Visualization.PrepareLines(drops)
    trailLines.Update(trailData)


if displaySurface:
    print('Animation is done')
    mainWindow.Keep(True)


'''
tic = time.time()
for i in range(0,10,1):
    drop = WDA.WaterDrop(mapSize)
    drop.LinkToHeightMap(heightMap)
    for j in range(0,64, 1):
        drop.Move()
print(drop)
toc = time.time()
print('elapsed time : %s' % (toc - tic))
'''

'''
a = Visualization.Window3D(xLim = [0, mapSize], yLim = [0, mapSize], zLim = [0, initialMaximumHeight], view = 'custom')


# Choose 'custom' or 'topdown' as view options.
#visualizerObject = Visualization.SurfaceVisualizer(xLim = [0, mapSize], yLim = [0, mapSize], zLim = [0, initialMaximumHeight],
#                                                   view = 'custom')
b = Visualization.Surf(a)
c = Visualization.Plot3(a)
b.Update(heightMap)

print('Visualizer has been created')
for i in range(0, 100, 1):
    #heightMap /= 1.1
    #heightMap = Noise.SimpleNoise(mapSize, 3)
    #heightMap *= initialMaximumHeight
    #visualizerObject.Update(heightMap)
    #drop.Move()
    #
    #c.Update(drop.x, drop.y, drop.z)
    print('hej')
'''

#for i in range(numberOfSteps)
    #






"""
============
3D animation
============

A simple example of an animated plot... In 3D!

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def Gen_RandLine(length, dims=2):

    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_RandLine(25, 3) for index in range(2)]


# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]


# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')


#for i in range(25):
#    update_lines(i, data, lines)
#    plt.pause(0.1)


# Creating the Animation object
#line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                                   interval=50, blit=False)

#plt.show()
"""




print('The program has ended')
#visualizerObject.Keep(False) # The visualizer window should not be kept after the program has finished executing
#a.Keep(True)


#
# The waterdrops should be able to be plotted along side with the surface.
# Make a ParticleVisualizer
#
