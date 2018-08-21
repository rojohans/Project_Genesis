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


'''

import Simulation.Erosion.Hydrolic.WDA as WDA
import Simulation.Noise as Noise
import Visualization
import numpy as np
import time # Used to time code, like tic-toc.
import cProfile


mapSize = 512
initialMaximumHeight = 100
numberOfRuns = 10
numberOfDrops = 1000 # THIS NEEDS TO BE 1 TO PREVENT -INF HOLES.
numberOfSteps = 64

displaySurface = True
displayTrail = False
performProfiling = False


maximumErosionRadius = 10  # This determines how many erosion templates should be created.


# The height map is generated from "simple noise".
heightMap = Noise.SimpleNoise(mapSize,2,2)
heightMap *= initialMaximumHeight
print('Noise has been generated')


# Create visualization objects
if displaySurface:
    initialWindow = Visualization.Window3D(xLim = [0, mapSize],
                                           yLim = [0, mapSize],
                                           zLim = [0, initialMaximumHeight],
                                           view ='topdown')
    mainWindow = Visualization.Window3D(xLim = [0, mapSize],
                                        yLim = [0, mapSize],
                                        zLim = [0, initialMaximumHeight],
                                        view ='topdown')
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

print('Amount of material before simulation: %s' % np.sum(heightMap))
tic = time.time()
for iRun in range(numberOfRuns):
    # Create the drops
    drops = [WDA.WaterDrop(mapSize,
                           numberOfSteps=numberOfSteps,
                           storeTrail=displayTrail,
                           inertia=0.7,
                           capacityMultiplier=30,
                           depositionRate=0.9,
                           erosionRate=0.1,
                           erosionRadius=4) for index in range(numberOfDrops)]
    WDA.WaterDrop.LinkToDrops(drops)
    WDA.WaterDrop.heightMapChange = np.zeros((mapSize, mapSize))
    WDA.WaterDrop.upperDepositionLimit = 20
    #WDA.WaterDrop.upperErosionLimit = # Not used yet.


    # Move and animate the drops
    for iStep in range(numberOfSteps):
        #print(len(drops))
        for drop in drops:
            drop()
    for drop in drops:
        drop.Deposit(depositAll=True)
    heightMap += WDA.WaterDrop.heightMapChange
toc = time.time()
print('elapsed time : %s sec' % (toc - tic))
print('Amount of material after simulation: %s' % np.sum(heightMap))

print(np.min(WDA.WaterDrop.heightMapChange))
print(np.max(WDA.WaterDrop.heightMapChange))
print(np.sum(WDA.WaterDrop.heightMapChange))

print(np.min(heightMap))
print(np.max(heightMap))


if performProfiling:
    pr.disable()
    pr.print_stats(2)


if displaySurface:
    mapSurface.Update(heightMap)
    #mapSurface.Update(WDA.WaterDrop.heightMapChange)


# The trails of the drops are visualized.
if displayTrail:
    trailData = Visualization.PrepareLines(drops)
    trailLines.Update(trailData)


if displaySurface:
    print('Animation is done')
    mainWindow.Keep(True)



print('The program has ended')
