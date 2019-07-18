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
import Simulation.FluidDynamics as FlDyn  # Be aware that this needs to change if the folder structure were to change,
import Simulation.Noise as Noise
import Visualization
import numpy as np
import time  # Used to time code, like tic-toc.
import cProfile

# =================================================
#                    SETTINGS
# =================================================
mapSize = 512
initialMaximumHeight = 300
numberOfRuns = 1000
numberOfDrops = 1  # This do not need to be 1 but changing it does not result in true parallel drops.
numberOfSteps = 100
waterCalculationFrequency = 2000  # The number of drops simulated between each set of flood/spill calculations.
maximumErosionRadius = 10  # This determines how many erosion templates should be created.

# http://eveliosdev.blogspot.com

inertia = 0.7
capacityMultiplier = 40
depositionRate = 0.01
erosionRate = 0.01
erosionRadius = 4
evaporationRate = 0.02
maximumUnimprovedSteps = 5

displaySurface = True
displayTrail = False  # This is not implemented in the mayavi version of the program.
performProfiling = False
saveWorld = False
PRINT_WATER_SEGMENT_LIST = False

# 100000  = 20 minuter
# 1000000 = 3 timmar
# 2000000 = 6 timmar


# The height map is generated from "simple noise".
heightMap = Noise.SimpleNoise(mapSize, 2, 2)
heightMap *= initialMaximumHeight

'''
heightMap = np.zeros((mapSize, mapSize))
for x in range(mapSize):
    for y in range(mapSize):
        heightMap[x, y] -= 1 / (1 + 0.1 * ((x - mapSize / 2) ** 2 + (y - mapSize / 2) ** 2)**(1.8))
        heightMap[x, y] -= 1 / (1 + 0.1 * ((x - 2 * mapSize / 3) ** 2 + (y - mapSize / 2) ** 2)**(1.4))
        heightMap[x, y] -= 1 / (1 + 0.1 * ((x - mapSize / 2) ** 2 + (y - 2 * mapSize / 3) ** 2)**(1))
        heightMap[x, y] -= 1 / (1 + 0.1 * ((x - 1 * mapSize / 3) ** 2 + (y - mapSize / 2) ** 2)**(0.6))
        heightMap[x, y] -= 1 / (1 + 0.1 * ((x - mapSize / 2) ** 2 + (y - 1 * mapSize / 3) ** 2)**(0.2))
'''

heightMap -= np.min(heightMap)
heightMap /= np.max(heightMap)
heightMap *= initialMaximumHeight

'''
# THIS CODE GENERATES INITIAL HILLS/MOUNTAINS
heightMap = np.zeros((mapSize, mapSize))
for x in range(mapSize):
    for y in range(mapSize):
        heightMap[x, y] += 1/(1 + 0.0002*((x-mapSize/2)**2 + (y-mapSize/2)**2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 3) ** 2 + (y - mapSize / 1.5) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 1.2) ** 2 + (y - mapSize / 3.1) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 1.2) ** 2 + (y - mapSize / 1.8) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 2.3) ** 2 + (y - mapSize / 2.8) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 2.9) ** 2 + (y - mapSize / 3.6) ** 2))
heightMap -= np.min(heightMap)
heightMap[heightMap < 0.2] = 0.2
a = Noise.SimpleNoise(mapSize,3,1.1)
a *= 0.2
heightMap += a
heightMap -= np.min(heightMap)
heightMap /= np.max(heightMap)
heightMap *= initialMaximumHeight
'''

initialRockMap = heightMap.copy()  # Used to determine where sediment has accumulated.
initialSedimentMap = np.zeros((mapSize, mapSize))
initialTotalMap = heightMap.copy()
initialWaterMap = np.zeros((mapSize, mapSize))

rockMap = initialRockMap.copy()
sedimentMap = initialSedimentMap.copy()
totalHeightMap = heightMap.copy() + sedimentMap.copy()
waterMap = initialWaterMap.copy()  # Contains the depth of water for each cell.

# Contains the ID value of each water segment. None symbolizes that there are no segment- or border-cells in those
# specific cells. A positive value indicates a specific segment ID, a negative indicates a specific border for the
# corresponding ID.
waterSegmentIdentificationMap = [None for i in range(mapSize ** 2)]
print('Noise has been generated')

# Creates a mayavi window and visualizes the initial terrain.
window = Visualization.MayaviWindow()
window.Surf(initialRockMap, type='terrain', scene='original')

waterSegments = []  # A list which will store the segment objects created.
drops = []

# Links the FluidSegment class to the heightmaps. These maps are used to determine how the water flows.
FlDyn.FluidSegment.LinkToHeightMap(rockMap, sedimentMap, totalHeightMap, waterMap, waterSegmentIdentificationMap)
FlDyn.FluidSegment.LinkToWaterSegments(waterSegments)
FlDyn.FluidSegment.LinkToDropParameters(inertia=inertia,
                                        capacityMultiplier=capacityMultiplier,
                                        depositionRate=depositionRate,
                                        erosionRate=erosionRate,
                                        erosionRadius=erosionRadius,
                                        evaporationRate=evaporationRate,
                                        maximumUnimprovedSteps=maximumUnimprovedSteps,
                                        maximumNumberOfSteps=numberOfSteps)

# Creates templates used by all the drops.
WDA.WaterDrop.LinkToHeightMap(rockMap, sedimentMap, totalHeightMap, waterMap, waterSegmentIdentificationMap)
WDA.WaterDrop.InitializeErosionTemplates(maximumErosionRadius)
WDA.ContinuousDrop.InitializeAdjacentTileTemplate()
WDA.DiscreteDrop.InitializeAdjacentTileTemplate()
WDA.ContinuousDrop.LinkToWaterSegments(waterSegments)

if performProfiling:
    pr = cProfile.Profile()
    pr.enable()

print('Amount of material before simulation: ', np.sum(rockMap) + np.sum(sedimentMap))
tic = time.time()




def Periodic(value):
    return (value + mapSize) % mapSize


adjacentTilesTemplate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

erosionWeightTemplate = []
erosionTileTemplate = []
for radius in range(0, 10):
    rows = range(-radius, radius, 1)
    columns = range(-radius, radius, 1)
    rowGrid, columnGrid = np.meshgrid(rows, columns)
    rowList = np.reshape(rowGrid, [rowGrid.size, 1])
    columnList = np.reshape(columnGrid, [columnGrid.size, 1])
    distances = np.sqrt(rowList ** 2 + columnList ** 2)
    rowList = rowList[distances < radius]
    columnList = columnList[distances < radius]

    # The template lists are expanded.
    erosionWeightTemplate.append(
        (radius - distances[distances < radius]) / np.sum(radius - distances[distances < radius]))

    erosionTiles = np.zeros((erosionWeightTemplate[radius].shape[0], 2))
    erosionTiles[:, 0] = rowList
    erosionTiles[:, 1] = columnList
    erosionTileTemplate.append(erosionTiles)

for iRun in range(numberOfRuns):
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #                              DROP RESET
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    xPreviousStep = None
    yPreviousStep = None
    zPreviousStep = None
    adjacentTilesPrevious = None  # Used for deposition.
    adjacentHeightsPrevious = None  # Used for deposition.
    heightDifference = None
    direction = np.array([0, 0])
    direction = adjacentTilesTemplate[np.random.randint(0, 4), :]  # initially a random direction is choosen.

    unimprovedSteps = 0
    maximumUnimprovedSteps = maximumUnimprovedSteps
    numberOfUnimprovedSteps = 0
    step = 0  # The number of steps taken
    maximumNumberOfSteps = numberOfSteps  # The amount of steps taken before termination

    velocity = 0
    waterAmount = 10
    sedimentCapacity = 0
    sedimentAmount = 0

    inertia = inertia
    gravity = 10
    capacityMultiplier = capacityMultiplier
    minimumSlope = 0.005
    depositionRate = depositionRate
    erosionRate = erosionRate
    erosionRadius = erosionRadius
    evaporationParameter = 1 + evaporationRate


    # If x and y values are not given they will be randomized to be within the grid.
    x = mapSize * np.random.rand()
    y = mapSize * np.random.rand()

    xWholePart = np.floor(x)
    yWholePart = np.floor(y)
    xDecimalPart = x - xWholePart
    yDecimalPart = y - yWholePart

    adjacentTiles = np.zeros((4, 2))
    adjacentTiles[:, 0] = Periodic(adjacentTilesTemplate[:, 0] + yWholePart)
    adjacentTiles[:, 1] = Periodic(adjacentTilesTemplate[:, 1] + xWholePart)
    adjacentHeights = rockMap[
        adjacentTiles[:, 0].astype(int), adjacentTiles[:, 1].astype(int)]

    # Interpolates the z coordinate value of the drop based on the surrounding heights.
    z = (1 - xDecimalPart) * (adjacentHeights[0] * (1 - yDecimalPart) + adjacentHeights[2] * yDecimalPart) \
             + xDecimalPart * (adjacentHeights[1] * (1 - yDecimalPart) + adjacentHeights[3] * yDecimalPart)


    while(step < maximumNumberOfSteps and unimprovedSteps < maximumUnimprovedSteps):
        # self.StoreState()
        # self.Move()
        # self.UpdateVelocity()
        # self.DepositOrErode()
        # self.CheckForTermination()
        # self.Evaporate()

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                            STATE STORAGE
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # The coordinates are stored for later use.
        xPreviousStep = x
        yPreviousStep = y
        zPreviousStep = z
        # Stores the adjacentTiles and heights for later use when doing deposition.
        adjacentTilesPrevious = adjacentTiles
        adjacentHeightsPrevious = adjacentHeights

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                            MOVE
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # The gradient is computed. If necessary the gradient can be randomized in the case that it's a null vector.
        xGradient1 = adjacentHeights[1] - adjacentHeights[0]
        xGradient2 = adjacentHeights[3] - adjacentHeights[2]
        yGradient1 = adjacentHeights[2] - adjacentHeights[0]
        yGradient2 = adjacentHeights[3] - adjacentHeights[1]
        gradient = np.array([xGradient1*(1-yDecimalPart) + xGradient2*yDecimalPart, \
                             yGradient1 * (1 - xDecimalPart) + yGradient2 * xDecimalPart])

        # If the gradient is a null vector a random unit vector is choosen.
        gradientNorm = np.linalg.norm(gradient)
        if gradientNorm == 0:
            randomAngle = 2*np.pi * np.random.rand()
            gradient[0] = np.cos(randomAngle)
            gradient[1] = np.sin(randomAngle)
        else:
            gradient /= gradientNorm
            #gradient /= np.linalg.norm(gradient)

        # The direction of the water drop is updated. This should depend on the drops inertia (velocity and mass), right
        #  now a constant inertia is used.
        #self.direction = self.direction * self.inertia - gradient * (1-self.inertia)
        direction = direction * inertia * velocity**2 - gradient * (1-inertia)
        direction /= np.linalg.norm(direction)

        # The drop is moved
        x += direction[0]
        y += direction[1]

        # The whole and decimal parts of the coordinates are updated
        xWholePart = np.floor(x)
        yWholePart = np.floor(y)
        xDecimalPart = x - xWholePart
        yDecimalPart = y - yWholePart

        # The adjacent tiles are selected using the adjacent tile template.
        adjacentTiles = np.zeros((4, 2))
        adjacentTiles[:, 0] = Periodic(adjacentTilesTemplate[:, 0] + yWholePart)
        adjacentTiles[:, 1] = Periodic(adjacentTilesTemplate[:, 1] + xWholePart)
        # Retrieves the map height for the adjacent tiles and interpolates the height of the water drop.
        adjacentHeights = rockMap[adjacentTiles[:, 0].astype(int), adjacentTiles[:, 1].astype(int)]

        # Interpolates the z coordinate value of the drop based on the surrounding heights.
        z = (1-xDecimalPart) * (adjacentHeights[0]*(1-yDecimalPart) + adjacentHeights[2]*yDecimalPart) \
            + xDecimalPart * (adjacentHeights[1] * (1-yDecimalPart) + adjacentHeights[3]*yDecimalPart)
        heightDifference = z - zPreviousStep

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                            UPDATE VELOCITY
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # If the drop were to go uphill it loses all it's velocity (!!!THIS SHOULD BE MODIFIED!!!)
        if heightDifference>0:
            velocity /= 2
        else:
            velocity = velocity/2 + np.arctan(-heightDifference) * gravity/np.pi

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                            DEPOSIT OR ERODE
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Determines if erosion or deposition should occur.
        if heightDifference > 0:
            heightDifferences = z - adjacentHeightsPrevious
            tilesToDeposit = adjacentTilesPrevious[heightDifferences > 0]
            heightDifferences = heightDifferences[heightDifferences > 0]

            if np.sum(heightDifferences) > sedimentAmount:
                # All sediment is deposited. (The hole can not be completely filled.)
                depositionAmount = sedimentAmount
            else:
                # Only a part of the sediment is deposited. (Enough to fill the hole.)
                depositionAmount = np.sum(heightDifferences)

            rockMap[tilesToDeposit[:, 0].astype(int), tilesToDeposit[:, 1].astype(int)] += \
                depositionAmount * heightDifferences / np.sum(heightDifferences)
            sedimentAmount -= depositionAmount

        else:
            #self.sedimentCapacity = np.max((-self.heightDifference, self.minimumSlope)) * self.velocity * \
            #                        self.waterAmount * self.capacityMultiplier
            # The sediment capacity is capped by the amount of water. Idealy It should be capped to a percentage,
            # 30% for example, of the water amount.
            #self.sedimentCapacity = np.min((self.waterAmount, np.max((-self.heightDifference, self.minimumSlope)) * self.velocity * self.capacityMultiplier))
              #self.sedimentCapacity = np.max((-self.heightDifference, self.minimumSlope)) * self.velocity * self.waterAmount * self.capacityMultiplier
            sedimentCapacity = velocity * waterAmount * capacityMultiplier
            #print(self.sedimentCapacity, np.max((-self.heightDifference, self.minimumSlope)), self.velocity, self.waterAmount)
            #self.sedimentCapacity = self.capacityMultiplier * np.min((self.waterAmount, np.max((-self.heightDifference, self.minimumSlope)) * self.velocity))
            #self.sedimentCapacity = np.max((-self.heightDifference, self.minimumSlope)) * self.velocity * self.capacityMultiplier * self.waterAmount

            if sedimentAmount < sedimentCapacity:
                # Select tiles to erode using templates.
                xWholePart = np.floor(xPreviousStep)
                yWholePart = np.floor(yPreviousStep)
                tilesToErode = np.zeros((erosionTileTemplate[erosionRadius].shape[0], 2))
                tilesToErode[:, 0] = Periodic(erosionTileTemplate[erosionRadius][:, 0] + yWholePart)
                tilesToErode[:, 1] = Periodic(erosionTileTemplate[erosionRadius][:, 1] + xWholePart)

                # Determines the amount of material to remove.
                erosionAmount = np.min(
                    ((sedimentCapacity - sedimentAmount) * erosionRate, - heightDifference))
                erosionWeights = erosionWeightTemplate[erosionRadius]

                # There is a minor problem in this part, it is possible to remove more sediment than there is sediment present.
                # This problem should be very minor until the weathering process is implemented.

                # Material is removed from the map and sediment is added to the drop.

                rockMap[tilesToErode[:, 0].astype(int), tilesToErode[:, 1].astype(int)] -= \
                    erosionAmount * erosionWeights
                sedimentAmount += erosionAmount

            else:
                heightDifferences = z - adjacentHeightsPrevious
                tilesToDeposit = adjacentTilesPrevious[heightDifferences > 0]
                heightDifferences = heightDifferences[heightDifferences > 0]

                depositionAmount = (sedimentAmount - sedimentCapacity) * depositionRate

                rockMap[tilesToDeposit[:, 0].astype(int), tilesToDeposit[:, 1].astype(int)] += \
                    depositionAmount * heightDifferences / np.sum(heightDifferences)
                sedimentAmount -= depositionAmount

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                            CHECK FOR IMPROVEMENT
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        if heightDifference > 0:
            unimprovedSteps += 1
        else:
            unimprovedSteps = 0

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        #                                EVAPORATE
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        waterAmount /= evaporationParameter




        step +=1

    #print(step)
    #print(' ')













        #print(iRun)





    #
toc = time.time()
print('elapsed time : %s sec' % (toc - tic))
print('Amount of material after simulation: ', np.sum(rockMap) + np.sum(sedimentMap))

if performProfiling:
    pr.disable()
    pr.print_stats(1)

if saveWorld:
    import Storage
    import pickle
    import datetime

    now = datetime.datetime.now()  # Date and time used as filename.
    # Creates a World object. It contains the heightmaps generated and the parameters used in the process.
    world = Storage.World(initialRockMap,
                          initialSedimentMap,
                          initialTotalMap,
                          initialWaterMap,
                          rockMap,
                          sedimentMap,
                          totalHeightMap,
                          waterMap,
                          numberOfRuns,
                          numberOfDrops,
                          numberOfSteps,
                          inertia,
                          capacityMultiplier,
                          depositionRate,
                          erosionRate,
                          erosionRadius,
                          maximumUnimprovedSteps)
    pickle.dump(world, open('Worlds/' + now.strftime("%Y-%m-%d %H:%M") + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    from PIL import Image

    snapshotMap = 255 * totalHeightMap / np.max(totalHeightMap)
    # np.repeat()
    img = Image.fromarray(snapshotMap.astype('uint8'), 'L')  # astype('uint8')  'L'
    # img.show()
    # img = img.rotate(90) # The rotation is necessary for the image to align with the surface properly.
    # img = img.transpose(Image.TRANSPOSE)  # The rotation is necessary for the image to align with the surface properly.
    img.save('Worlds/' + now.strftime("%Y-%m-%d %H:%M") + '.png')

if PRINT_WATER_SEGMENT_LIST:
    for waterSegment in waterSegments:
        print('Segment ID: ', waterSegment.ID)
        print('Total fluid amount: ', waterSegment.totalFluidAmount)
        print('Available fluid: ', waterSegment.availableFluid)
        print('')

# Visualizes the eroded terrain.
if displaySurface:
    # window.Surf(totalHeightMap, type='terrain', scene='original')
    window.Surf(rockMap, type='terrain', scene='updated')
    # sedimentMap = heightMap-unchangedHeightMap
    # sedimentMap[sedimentMap > 0] = 0
    # heightMapToPlot = unchangedHeightMap + sedimentMap
    # window.Surf(heightMapToPlot, type='terrain', scene='updated')

    # sedimentMap = heightMap-unchangedHeightMap
    # sedimentMap[sedimentMap < 0] = 0
    # c = unchangedHeightMap + sedimentMap
    # c[sedimentMap == 0] = 0


    # window.Surf(30+np.zeros([mapSize, mapSize]), type='water', scene='updated')
    # window.Surf(c, type='water', scene='updated')

    a = rockMap.copy()
    a[waterMap == 0] = 0
    window.Surf(a + waterMap, type='water', scene='updated')
    print('Maximum water depth: ', np.max(waterMap))
    print('Total amount of water: ', np.sum(waterMap))

    # np.min(sedimentMap)
    rockMap[sedimentMap == 0] = 0
    window.Surf(rockMap + sedimentMap, type='sediment', scene='updated')
    # window.SedimentColour(sedimentMap)

    window.configure_traits()
    # mapSurface.Update(heightMap)

# The trails of the drops are visualized.
if displayTrail:
    pass
    # trailData = Visualization.PrepareLines(drops)
    # trailLines.Update(trailData)

if displaySurface:
    pass
    # print('Animation is done')
    # mainWindow.Keep(True)





