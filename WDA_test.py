'''
From this sqript file the waterdrop algorithm can be tested.

The program will create a severe bug if parallell drops are used (numberOfDrops >1).
    # In the case of parallell drops the adjacentHeights and the heightDifference needs to be recalculated, these
    # values depends on the heightMap which could have been changed by another drop. If this is not taken into
    # account multiple drops can erode the same area creating a deep hole. (the depth of the hole will approach -inf).
    # An alternative approach is to limit the amount of deposition/erosion to 1/N of the highest allowed
    # deposition/erosion value for each drop. N is the number of drops, in that case it would require all N drops
    # to deposit/erode all the material, thusly no holes or peaks could be formed. A downside is that it lowers the
    # effect of each drop, thusly requiring more drops to get the same effect. If the code utilizes a GPU for
    # parallell computations this approach could be viable.


    The inertia part of the DiscreteDrop class needs to be altered. Perhaps the previous direction should be included when choosing new direction, using the inertia as weights.

    When choosing "topdown" in the visualizer the plot object generated should not be a 3D surface.
'''

import Simulation.Erosion.Hydrolic.WDA as WDA
import Simulation.Noise as Noise
import Visualization
import numpy as np
import time # Used to time code, like tic-toc.
import cProfile


mapSize = 512
initialMaximumHeight = 100
numberOfRuns = 1
numberOfDrops = 1 # This do not need to be 1 but. Changing it does not result in parallel drops.
numberOfSteps = 64
maximumErosionRadius = 10  # This determines how many erosion templates should be created.

displaySurface = False
displayTrail = False
performProfiling = False


# The height map is generated from "simple noise".
heightMap = Noise.SimpleNoise(mapSize,2,2)
heightMap *= initialMaximumHeight
print('Noise has been generated')


# Create visualization objects
# Choose 'custom' or 'topdown' as view option for the Window3D objects.
# The xPosition and yPosition values are to be given in pixels. A window with position (0, 0) is located in the upper
# left corner. The width and height values are given in inches. 1 inch = 2.54 cm. 1 inch = 200 pixels (Robin's Laptop).
if displaySurface:
    initialWindow = Visualization.Window3D(xPosition = 900,
                                           yPosition = 70,
                                           width = 5,
                                           height = 5,
                                           xLim = [0, mapSize],
                                           yLim = [0, mapSize],
                                           zLim = [0, initialMaximumHeight],
                                           view ='topdown')
    mainWindow = Visualization.Window3D(xPosition = 400,
                                        yPosition = 70,
                                        width = 5,
                                        height = 5,
                                        xLim = [0, mapSize],
                                        yLim = [0, mapSize],
                                        zLim = [0, initialMaximumHeight],
                                        view ='custom')
    initialMapSurface = Visualization.Surf(initialWindow, z = heightMap)
    mapSurface = Visualization.Surf(mainWindow, z = heightMap)
    waterSurface = Visualization.Surf(mainWindow, z=5+np.zeros([mapSize, mapSize]), visibility = 0.5)
    if displayTrail:
        trailLines = Visualization.Lines(mainWindow, numberOfDrops)


# Creates templates used by all the drops.
WDA.WaterDrop.LinkToHeightMap(heightMap)
WDA.WaterDrop.InitializeErosionTemplates(maximumErosionRadius)
WDA.ContinuousDrop.InitializeAdjacentTileTemplate()
WDA.DiscreteDrop.InitializeAdjacentTileTemplate()


if performProfiling:
    pr = cProfile.Profile()
    pr.enable()

print('Amount of material before simulation: %s' % np.sum(heightMap))
tic = time.time()
for iRun in range(numberOfRuns):
    # Create the drops
    drops = [WDA.ContinuousDrop(
                           numberOfSteps=numberOfSteps,
                           storeTrail=displayTrail,
                           inertia=0.1,
                           capacityMultiplier=200,
                           depositionRate=0.1,
                           erosionRate=0.01,
                           erosionRadius=4,
                           maximumUnimprovedSteps = 5) for index in range(numberOfDrops)]
    WDA.WaterDrop.LinkToDrops(drops)
    # Performs the drop simulation, step by step.
    for iStep in range(numberOfSteps):
        for drop in drops:
            drop()


toc = time.time()
print('elapsed time : %s sec' % (toc - tic))
print('Amount of material after simulation: %s' % np.sum(heightMap))


print(np.min(heightMap))
print(np.max(heightMap))


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
from mayavi import mlab
import numpy as np

x, y, z = np.mgrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x*y*z)/(x*y*z)

mlab.pipeline.volume(mlab.pipeline.scalar_field(x,y,z,s))
mlab.show()
'''


#from mayavi import mlab
#mlab.show()









'''
import numpy
import gnuplot

def rainfall_intensity_t10(t):
    return 11.23 * (t**(-0.713))

def rainfall_intensity_t50(t):
    return 18.06 * (t**(-0.713))

g = gnuplot.gnuplot()
g.title("rainfall intensity")

g.xlabel("t (min)")
g.ylabel("i (mm/min)")

g("set grid")
g("set xtic 10")
g("set ytic 1")

x = numpy.arange (start=2, stop=120, step=0.5, dtype='float_')

y1 = rainfall_intensity_t10(x) # yields another numpy.arange object
y2 = rainfall_intensity_t50(x) # ...

d1 = gnuplot.Data (x, y1, title="intensity i (T=10)", with_="lines")
d2 = gnuplot.Data (x, y2, title="intensity i (T=50)", with_="lines")

g("set terminal svg")
g.plot(d1, d2) # write SVG data directly to stdout ...
'''






from mayavi import mlab
from tvtk.api import tvtk

#help(mlab.figure)

fig = mlab.figure(size=(700, 700))
fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

mlab.surf(30+np.zeros([mapSize, mapSize]),
          opacity = 0.5,
          color = (0.1, 0.3, 0.7))
mlab.surf(heightMap,
          color = (0.6, 0.4, 0.3))



secondFigure = mlab.figure()
fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

mlab.surf(heightMap,
          color = (0.6, 0.4, 0.3))


mlab.show()




'''
from mayavi import mlab
from mayavi.sources.builtin_surface import BuiltinSurface

from tvtk.api import tvtk
fig = mlab.gcf()
fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


ocean_blue = (0.4, 0.5, 1.0)
r = 6371 # km

sphere = mlab.points3d(0, 0, 0, name='Globe',
  scale_mode='none', scale_factor=r * 2.0,
  color=ocean_blue, resolution=50)

sphere.actor.property.specular = 0.20
sphere.actor.property.specular_power = 10

continents_src = BuiltinSurface(source='earth', name='Continents')
continents_src.data_source.on_ratio = 1  # detail level
continents_src.data_source.radius = r
continents = mlab.pipeline.surface(continents_src, color=(0, 0, 0))

mlab.show()
'''



'''
from mayavi import mlab
mlab.figure(bgcolor=(0.1,0.5,0.8))
#mlab.points3d(0,0,0,color=(1,0,0))
mlab.test_surf()
#mlab.test_molecule()
#mlab.test_plot3d()
mlab.show()
'''




'''
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input_data

mlab.options.offscreen = True
#mlab.test_plot3d()
mlab.test_surf()
#mlab.test_molecule()
fig = mlab.gcf()
rw = tvtk.RenderWindow(size=fig.scene._renwin.size, off_screen_rendering=1)
rw.add_renderer(fig.scene._renderer)

w2if = tvtk.WindowToImageFilter()
w2if.magnification = fig.scene.magnification
w2if.input = rw
ex = tvtk.PNGWriter()
ex.file_name = 'example4.png'
configure_input_data(ex, w2if.output)
w2if.update()
ex.write()
'''



'''
from mayavi import mlab
#mlab.points3d(1,1,1)
mlab.test_plot3d()
#mlab.savefig('testImage.jpg')
mlab.show()
'''


'''
# Create the data.
from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z)
mlab.show()
'''





'''
import vtk
print('vtk ok')

import mayavi
print('mayavi ok')

import mayavi.mlab
print('mlab ok')
'''

'''
#import numpy
from mayavi import mlab
import mayavi
def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    print(f(x, y))
    s = mlab.surf(x, y, f(x, y))
    mlab.draw()
    mlab.show()
    #cs = contour_surf(x, y, f, contour_z=0)
    return s

x = test_surf()
print(x)
'''






'''
# library
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Get the data (csv file is hosted on the web)
url = 'https://python-graph-gallery.com/wp-content/uploads/volcano.csv'
data = pd.read_csv(url)

# Transform it to a long format
df = data.unstack().reset_index()
df.columns = ["X", "Y", "Z"]

# And transform the old column name in something numeric
df['X'] = pd.Categorical(df['X'])
df['X'] = df['X'].cat.codes

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)


for angle in range(0,360,5):
    ax.view_init(30, angle)
    plt.pause(0.000001)


#ax.view_init(30, 45)



plt.show()
'''


'''
# to Add a color bar which maps values to colors.
surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# Rotate it
ax.view_init(30, 45)
plt.show()

# Other palette
ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
plt.show()
'''










print('The program has ended')
