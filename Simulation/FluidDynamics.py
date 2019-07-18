import numpy as np
from .Erosion.Hydrolic import WDA


# THIS MODULE SHALL CONTAIN METHODS REGARDING FLUID DYNAMICS

class FluidSegment():
    '''
    interiorCells: These are the cells which are filled with fluid
    borderCells: These are the cells which surrounds the outer most interior cells.

    ########## FLUID-FILL-ALGORITHM
    - Get adjacent cells.

    - Sort adjacent cells.

    - If there is an adjacent cell with lower elevation than the water level -> STOP, if not pick the cell with the
      lowest elevation. In the case of STOP a new drop should be created. It shall contain all remaining available
      water, it should be located at the newly created segment-piece with a direction pointing away from the center
      of the segment.

    - That cell is now part of the segment and it's elevation will determine the water level, if there is sufficient
      water available. If there is not enough water to completely fill to the new water level -> STOP.
    '''

    numberOfWaterSegmentsCreated = 0

    # An array used to determine which cells should be classified as the adjacent cells.
    adjacencyTemplate = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    #adjacencyTemplate = [(-1, 0), (1, 0), (0, -1), (0, 1)] # List of tuples.

    def __init__(self, totalFluidAmount, initialRow, initialColumn):
        # The fluidAmount is the amount of fluid in each cell. Initialy all the fluid is concentrated in one cell.

        # Assigns a specific ID value to the created water segment.
        self.ID = self.numberOfWaterSegmentsCreated
        FluidSegment.numberOfWaterSegmentsCreated += 1

        #print('Initial cell from GetStartCell()', [initialRow, initialColumn])
        [initialCell, self.fluidElevation] = self.GetLowestCell(initialRow, initialColumn)
        #print('Initial cell from GetLowestCell(): ', initialCell)
        # self.cells = [(initialRow, initialColumn)]
        #self.interiorCells = [np.array([[initialRow, initialColumn]])]
        self.waterSegmentIdentificationMap[initialCell[0, 0].astype(int) + initialCell[0, 1].astype(int) * self.gridSize] = self.ID
        self.interiorCells = [initialCell]
        self.fluidAmount = [0] # The amount of fluid in each cell.
        self.totalFluidAmount = 0 # The total amount of fluid among all the cells.
        self.availableFluid = totalFluidAmount # The amount of fluid which has not yet been used to flood adjacent tiles.
        #self.fluidElevation = self.totalHeightMap[initialRow.astype(int), initialColumn.astype(int)]# This is the height value of the fluid
        # surface, it is not related to the depth of the fluid.

        # The borderCells are sorted from highest elevation (first element) to lowest elevation (last element).
        self.borderCells = []
        self.borderElevation = [] # The elevation value of the border cells.


    def InitiateFlow(self):
        '''
        It might be faster to add the adjacent cells to the border list and then sort the entire list instead of
        inserting the cells at the correct positions.
        :return:
        '''

        #print(' ')
        #print('segment ID: ', self.ID)
        #print('available fluid: ', self.availableFluid)
        #self.availableFluid = 1000

        # Used to prevent selection of border cells when getting unique adjacent cells.
        # False: No border in this cell
        # True: Border in this cell
        borderMap = [False for i in range(self.gridSize ** 2)]
        # Adds the previous border cells to the borderMap (local to this function call).
        for borderCell in self.borderCells:
            borderMap[borderCell[0, 0].astype(int) + borderCell[0, 1].astype(int) * self.gridSize] = True


        # "cell" should always be the newly added element of the list.
        #for cell in self.interiorCells:
        while self.availableFluid > 0:
            cell = self.interiorCells[-1]


            #<================ GETS THE ADJACENT CELLS ================>
            uniqueAdjacentCells = self.GetUniqueAdjacentCells(cell, borderMap)

            for aCell in uniqueAdjacentCells:
                if self.waterSegmentIdentificationMap[aCell[0, 0].astype(int) + aCell[0, 1].astype(int) * self.gridSize] is not None:
                    print('A different segment is on the border')
            #


            #<==================== GETS THE ELEVATION VALUES OF THE ADJACENT CELLS ====================>
            # Get elevation of the unique adjacent cells.
            uniqueAdjacentElevations = []
            for uniqueAdjacentCell in uniqueAdjacentCells:
                borderMap[uniqueAdjacentCell[0, 0].astype(int) + uniqueAdjacentCell[0, 1].astype(int)*self.gridSize] = True
                uniqueAdjacentElevations.append(self.totalHeightMap[uniqueAdjacentCell[0, 0].astype(int),
                                                                uniqueAdjacentCell[0, 1].astype(int)])


            # <=============== INSERT THE NEW ADJACENT CELLS, SORTED ==============>
            # This function uses a sorting algorithm to insert the new adjacent cells in the correct order in the border list.
            self.InsertNewBorderCells(uniqueAdjacentCells, uniqueAdjacentElevations)

            '''
            for aCell in uniqueAdjacentCells:
                print('adjacentCell: ', aCell)
                print(self.totalHeightMap[aCell[0, 0].astype(int),
                                    aCell[0, 1].astype(int)])
                self.waterSegmentIdentificationMap[
                    aCell[0, 0].astype(int) + aCell[0, 1].astype(int) * self.gridSize]
            #print(uniqueAdjacentCells)
            #print(uniqueAdjacentElevations)
            print('Fluid elevation: ', self.fluidElevation)
            '''

            # Checks if a lower cell has been found. If thas is the case all the remaining fluid should be used to create a new drop.
            # If a lower cell was not found the lowest of the adjacent cells should be used to set the new fluid
            # elevation level, the segment cells should than be filled to that level, if the is sufficient available fluid.
            if self.borderElevation[-1] < self.fluidElevation:
                print('Border elevation: ', self.borderElevation[-1])
                self.CreateNewDrop()
                return
            else:
                self.RaiseFluidLevel()


    def InsertNewBorderCells(self, uniqueAdjacentCells, uniqueAdjacentElevations):
        # The adjacent cells are sorted in ascending order.
        # Combines The coordinates and the elevations into one list, the list is sorted using the elevations and the
        # coordinates are returned. reverse = True makes the list sorted in descending order.
        uniqueAdjacentCells = [x for (y, x) in
                               sorted(zip(uniqueAdjacentElevations, uniqueAdjacentCells), key=lambda pair: pair[0],
                                      reverse=True)]  # Crashes if two adjacent cells have the same elevation value.
        uniqueAdjacentElevations = sorted(uniqueAdjacentElevations, reverse=True)

        # The initialComparisonindex is used to indicate where in the border list the comparison between the border
        # cells and the new cells should begin. The initialComparisonIndex reduces redundancies.
        initialComparisonIndex = 0
        # The adjacent cells is inserted into the border list at the proper location.
        for i in range(uniqueAdjacentCells.__len__()):
            uniqueAdjacentCell = uniqueAdjacentCells[i]
            uniqueAdjacentElevation = uniqueAdjacentElevations[i]
            earlyInsertion = False
            if self.borderCells.__len__() > 0:
                # Loops over cells in the border list.
                for j in range(initialComparisonIndex, self.borderCells.__len__()):
                    if uniqueAdjacentElevation > self.borderElevation[j]:
                        # The new cell is inserted at a specific location into the border list
                        self.borderCells.insert(j, uniqueAdjacentCell)
                        self.borderElevation.insert(j, uniqueAdjacentElevation)
                        earlyInsertion = True
                        initialComparisonIndex = j + 1
                        break
                if earlyInsertion is False:
                    # The new cell is placed at the bottom of the border list.
                    self.borderCells.append(uniqueAdjacentCell)
                    self.borderElevation.append(uniqueAdjacentElevation)
                    initialComparisonIndex = j + 1
            else:
                # If no border cells exist the choosen cell is made into the first border cell.
                self.borderCells = [uniqueAdjacentCell]
                self.borderElevation = [uniqueAdjacentElevation]

    def CreateNewDrop(self):
        # Updates the waterMap and the totalHeightMap.
        for specificCell, specificFluidAmount in zip(self.interiorCells, self.fluidAmount):
            self.waterMap[specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] = specificFluidAmount
            self.totalHeightMap[
                specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] += specificFluidAmount - \
                                                                                   self.waterMap[
                                                                                       specificCell[0, 0].astype(
                                                                                           int), specificCell[
                                                                                           0, 1].astype(int)]
        '''
        #       A NEW DROP SHOULD BE CREATED HERE
        '''
        print('LOWER ELEVATION FOUND')
        '''
        print('Current segment ID: ', self.ID)
        [a, b, waterSegmentID] = FluidSegment.GetStartCell(self.borderCells[-1][0, 0], self.borderCells[-1][0, 1])
        print('Lower segment ID', waterSegmentID)

        # Transfers fluid to the lower segment.
        self.waterSegments[waterSegmentID].availableFluid += self.availableFluid
        self.availableFluid = 0

        print(self.fluidElevation)
        print(self.waterSegments[waterSegmentID].fluidElevation)
        '''

        #self.availableFluid = 0
        return
        #print(self.waterSegments.__len__())

        print('Current segment ID: ', self.ID)

        [a, b, waterSegmentID] = FluidSegment.GetStartCell(self.borderCells[-1][0, 0], self.borderCells[-1][0, 1])

        # Check if the start cell is part of a segment or not. If it is fluid will be added to that segment, if not a
        # new segment will be created.
        if waterSegmentID is None:
            # SEGMENT IS CREATED
            print('Segment is created')
            print('availableFluid: ', self.availableFluid)
            createdWaterSegment = FluidSegment(self.availableFluid, a[0, 0], a[0, 1])
            createdWaterSegment.InitiateFlow()  # Makes the water flow and fill up adjacent tiles.

            # Adds the newly created water segment to the water segment list
            self.waterSegments.append(createdWaterSegment)
        else:
            # FLUID IS ADDED TO SEGMENT
            print('Segment is expanded')
            print('Segment to expand: ', waterSegmentID)
            encounteredSegmentID = waterSegmentID

            # This could be done using a better search algorithm. The better search algorithm would start
            # the search in the middle of the list and then divide the list into two parts. The next step is
            # to repeat the procedure until the correct object has been found. This is an efficient search
            # mechanism but can only be performed if the objects are sorted by their ID.
            # Gets the specific segment object for which the ID is known.
            for waterSegment in self.waterSegments:
                if encounteredSegmentID == waterSegment.ID:
                    waterSegment.availableFluid = self.availableFluid
                    waterSegment.InitiateFlow()
                    break
        print(' ')
        print(' ')




        '''
        [a, b] = self.GetLowestCell(self.borderCells[-1][0, 0], self.borderCells[-1][0, 1])

        #x = self.borderCells[-1][0, 1],
        #y = self.borderCells[-1][0, 0],
        spillOverDrop = WDA.ContinuousDrop(
                                        x = a[0, 1],
                                        y = a[0, 0],
                                        inertia=self.dropInertia,
                                        capacityMultiplier=self.dropCapacityMultiplier,
                                        depositionRate=self.dropDepositionRate,
                                        erosionRate=self.dropErosionRate,
                                        erosionRadius=self.dropErosionRadius,
                                        evaporationRate=self.dropEvaporationRate,
                                        maximumUnimprovedSteps=self.dropMaximumUnimprovedSteps,
                                        maximumNumberOfSteps=self.dropMaximumNumberOfSteps)
        spillOverDrop.waterAmount = self.availableFluid
        #a = self.borderCells[-1] - self.interiorCells[-1]
        #spillOverDrop.direction = np.array([a[0, 1], a[0, 0]])
        self.drops.append(spillOverDrop)

        print('Nearest lower cell: ', a)

        for cell in self.interiorCells:
            print('interior: ', cell, self.fluidElevation)
        for cell, elevation in zip(self.borderCells, self.borderElevation):
            print('border: ', cell, elevation)

        [a, b, c] = FluidSegment.GetStartCell(self.borderCells[-1][0, 0], self.borderCells[-1][0, 1])

        print('---')
        print(a)
        print(c)

        adjacentCells = self._Periodic(a + self.adjacencyTemplate)
        adjacentElevations = self.totalHeightMap[adjacentCells[:, 0].astype(int), adjacentCells[:, 1].astype(int)]

        print(adjacentCells)
        print(adjacentElevations)

        #exit()

        #print('Nearest lower cell: ', self.GetLowestCell(self.borderCells[-1][0, 0] , self.borderCells[-1][0, 1]))
        '''

        #b = np.array([a[0, 1], a[0, 0]])
        #print(a)
        #print(b)
        #exit()

        self.availableFluid = 0
        return

    def RaiseFluidLevel(self):
        # The difference between the old fluid level and the new, will determain how much fluid to add.
        fluidElevationDifference = self.borderElevation[-1] - self.fluidElevation  # Difference to fill.
        fluidAmountToNextLevel = np.size(self.fluidAmount) * fluidElevationDifference

        # CHecks how much fluid is available
        if fluidAmountToNextLevel < self.availableFluid:
            # All cells are completely filled to the next level.
            self.fluidAmount[:] += fluidElevationDifference  # Adds fluid to cells.
            self.fluidElevation = self.borderElevation[-1]  # The fluid level rises.
            self.totalFluidAmount += fluidAmountToNextLevel
            self.availableFluid -= fluidAmountToNextLevel

            self.fluidAmount.append(0)  # The water level of the newly added cell.
            self.interiorCells.append(self.borderCells[-1])
            del self.borderCells[-1]
            del self.borderElevation[-1]

            # Add the new interior cell to the identification map.
            self.waterSegmentIdentificationMap[
                self.interiorCells[-1][0, 0].astype(int) + self.interiorCells[-1][0, 1].astype(
                    int) * self.gridSize] = self.ID
        else:
            # All cells are filled with the remaining amount of available fluid.
            # np.float64() is used to parse a float value in order to avoid a bug (makes the code crash).
            self.fluidAmount[:] += np.float64(self.availableFluid / np.size(self.fluidAmount))
            self.fluidElevation += self.availableFluid / np.size(self.fluidAmount)
            self.totalFluidAmount += self.availableFluid
            self.availableFluid = 0

            # sumTmp = 0
            # for flA in self.fluidAmount:
            #    sumTmp += flA
            # print('Total amount of fluid: ', sumTmp)
            print('NOT ENOUGH FLUID')

            # Updates the waterMap and the totalHeightMap.
            for specificCell, specificFluidAmount in zip(self.interiorCells, self.fluidAmount):
                self.waterMap[
                    specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] = specificFluidAmount
                self.totalHeightMap[
                    specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] += specificFluidAmount - \
                                                                                       self.waterMap[
                                                                                           specificCell[
                                                                                               0, 0].astype(
                                                                                               int),
                                                                                           specificCell[
                                                                                               0, 1].astype(
                                                                                               int)]
                # self.waterMap[specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] += specificFluidAmount
                # self.totalHeightMap[specificCell[0, 0].astype(int), specificCell[0, 1].astype(int)] += specificFluidAmount
            #return


    def GetUniqueAdjacentCells(self, cell, borderMap):
        '''
        Returns a list containing the coordinates [row, column] of the adjacent cells to the input cell which are not
        part of the interior- or border cells.
        :param cell:
        :return:
        '''
        # The four cells around the center cell.
        adjacentCells = [self._Periodic(cell + self.adjacencyTemplate[i, :]) for i in range(4)]

        # The candidate cells which are not already part of the fluid cells(interior) or the border cells are selected.
        uniqueAdjacentCells = []
        for adjacentCell in adjacentCells:
            # [======][======][======][======][======][======][======][======][======]
            # [      The uniqueness check should be done using a look-up table       ]
            # [======][======][======][======][======][======][======][======][======]
            #self.waterSegmentIdentificationMap[adjacentCell[0, 0].astype(int) + adjacentCell[0, 1].astype(int)*self.gridSize]

            #notInInterior = next((False for elem in self.interiorCells if np.array_equal(elem, adjacentCell)), True)
            #notInBorder = next((False for elem in self.borderCells if np.array_equal(elem, adjacentCell)), True)
            #if notInInterior and notInBorder:
            if self.waterSegmentIdentificationMap[adjacentCell[0, 0].astype(int) + adjacentCell[0, 1].astype(int)*self.gridSize] is not self.ID and borderMap[adjacentCell[0, 0].astype(int) + adjacentCell[0, 1].astype(int)*self.gridSize] is False:
                # Border cells should be allowed.
                #print(adjacentCell)
                uniqueAdjacentCells.append(adjacentCell)
        return uniqueAdjacentCells


    @classmethod
    def LinkToHeightMap(cls, rockMap, sedimentMap, totalHeightMap, waterMap, waterSegmentIdentificationMap):
        '''
        This is used in order for the drop to have access to the heightmap. If the heightMap were to be changed by
        one drop all other drops would also notice the change. When assigning the gridSize the heightMap is assumed
        to be cubic in shape.
        :param rockMap:
        :return:
        '''
        cls.rockMap = rockMap
        cls.sedimentMap = sedimentMap
        cls.totalHeightMap = totalHeightMap
        cls.waterMap = waterMap
        cls.gridSize = np.shape(rockMap)[0]

        # Used to identify which tiles are part of which water segments
        cls.waterSegmentIdentificationMap = waterSegmentIdentificationMap

    @classmethod
    def LinkToDropParameters(cls, inertia = 0.3,
                                  capacityMultiplier = 200,
                                  depositionRate = 0.01,
                                  erosionRate = 0.5,
                                  erosionRadius = 3,
                                  evaporationRate = 0.02,
                                  maximumUnimprovedSteps = 3,
                                  maximumNumberOfSteps = 100):
        # Gives the Fluid Dynamics class access to drop specific parameters. These are used when new drops are created
        # as water segments "spill over".
        cls.dropInertia = inertia
        cls.dropCapacityMultiplier = capacityMultiplier
        cls.dropDepositionRate = depositionRate
        cls.dropErosionRate = erosionRate
        cls.dropErosionRadius = erosionRadius
        cls.dropEvaporationRate = evaporationRate
        cls.dropMaximumUnimprovedSteps = maximumUnimprovedSteps
        cls.dropMaximumNumberOfSteps = maximumNumberOfSteps

    @classmethod
    def LinkToWaterSegments(cls, waterSegments):
        cls.waterSegments = waterSegments

    @classmethod
    def LinkToDrops(cls, drops):
        cls.drops = drops

    @classmethod
    def _Periodic(cls, value):
        return (value + cls.gridSize) % (cls.gridSize)

    def GetLowestCell(self, row, column):
        # This method should return the coordinates of the nearest "hole". A hole is defined as a cell where all the
        # adjacent cells are of higher elevation.

        currentCell = np.array([[row, column]])

        #print(currentCell)

        #print('initialCell: ', currentCell)
        currentElevation = self.totalHeightMap[currentCell[0, 0].astype(int), currentCell[0, 1].astype(int)]

        while True:
            adjacentCells = self._Periodic(currentCell + self.adjacencyTemplate)
            adjacentElevations = self.totalHeightMap[adjacentCells[:, 0].astype(int), adjacentCells[:, 1].astype(int)]

            #print('current cell: ', currentCell)
            #print('current elevation: ', currentElevation)
            #print('adjacent cells: ', adjacentCells)
            #print('adjacent elevations: ', adjacentElevations)
            #print('---')

            # Select the lower adjacent tiles.
            adjacentCells = adjacentCells[adjacentElevations<currentElevation, :]
            adjacentElevations = adjacentElevations[adjacentElevations < currentElevation]

            # If there are no lower tiles the current tile should be returned, if not the process continues.
            if np.size(adjacentElevations) == 0:
                return [currentCell, currentElevation]
            else:
                currentCell = adjacentCells[adjacentElevations == np.min(adjacentElevations)]
                currentElevation = adjacentElevations[adjacentElevations == np.min(adjacentElevations)]

                # Neccesary if multiple cells has the same elevation values.
                if np.size(currentElevation) > 1:
                    #print('Several adjacent cells have the same elevation value.')

                    currentCell = np.array([[currentCell[0, 0], currentCell[0, 1]]])
                    currentElevation = currentElevation[0]

    @classmethod
    def GetStartCell(cls, row, column):
        # This method should return the coordinates of the nearest "hole". A hole is defined as a cell where all the
        # adjacent cells are of higher elevation.

        currentCell = np.array([[row, column]])

        #print(currentCell)

        # print('initialCell: ', currentCell)
        currentElevation = cls.totalHeightMap[currentCell[0, 0].astype(int), currentCell[0, 1].astype(int)]

        while True:
            adjacentCells = cls._Periodic(currentCell + cls.adjacencyTemplate)
            adjacentElevations = cls.totalHeightMap[adjacentCells[:, 0].astype(int), adjacentCells[:, 1].astype(int)]

            # Select the adjacent tiles of the same elevation.
            adjacentSegmentCells = adjacentCells[adjacentElevations == currentElevation]
            adjacentSegmentElevations = adjacentElevations[adjacentElevations == currentElevation]


            # Select the lower adjacent tiles.
            adjacentCells = adjacentCells[adjacentElevations < currentElevation, :]
            adjacentElevations = adjacentElevations[adjacentElevations < currentElevation]

            # If there are no lower tiles the current tile should be returned, if not the process continues.
            if np.size(adjacentElevations) == 0:
                if np.size(adjacentSegmentElevations) > 0:
                    pass
                    #for a in adjacentSegmentCells:
                        #print('Adjacent segment cell: ', a)
                        #print(a[0])
                        #print(a[1])
                        #print('Adjacent cell: ', a)
                        #print('Adjacent ID: ', cls.waterSegmentIdentificationMap[a[0].astype(int) + a[1].astype(int) * cls.gridSize])
                    #print(' ')
                return [currentCell, currentElevation, None]
            else:
                currentCell = adjacentCells[adjacentElevations == np.min(adjacentElevations)]
                currentElevation = adjacentElevations[adjacentElevations == np.min(adjacentElevations)]

                # Neccesary if multiple cells has the same elevation values.
                if np.size(currentElevation) > 1:
                    # print('Several adjacent cells have the same elevation value.')

                    currentCell = np.array([[currentCell[0, 0], currentCell[0, 1]]])
                    currentElevation = currentElevation[0]

            # Check if water has been reached. If so the function should returned the currentCell, its elevation and the
            # ID of the water segment found.
            if cls.waterSegmentIdentificationMap[currentCell[0, 0].astype(int) + currentCell[0, 1].astype(int)*cls.gridSize] is not None:

                #print(currentCell)
                #print(cls.waterSegmentIdentificationMap[
                #    currentCell[0, 0].astype(int) + currentCell[0, 1].astype(int) * cls.gridSize])
                #print(' ')
                return [currentCell, currentElevation, cls.waterSegmentIdentificationMap[currentCell[0, 0].astype(int) + currentCell[0, 1].astype(int)*cls.gridSize]]



'''
# Check if any border cells has been found. If no cells were found information about the current cell is to be printed.
            try:
                a = self.borderElevation[-1]
            except:
                #print(self.interiorCells)
                #print('available fluid: ', self.availableFluid)
                #print('segment ID: ', self.ID)
                #print(self.waterSegmentIdentificationMap[cell[0, 0].astype(int) + cell[0, 1].astype(int) * self.gridSize])
                adjacentCells = [self._Periodic(self.interiorCells[0] + self.adjacencyTemplate[i, :]) for i in range(4)]
                #for c in adjacentCells:
                    #print('adjacent cell: ', c)
                    #print('adjacent ID: ', self.waterSegmentIdentificationMap[c[0, 0].astype(int) + c[0, 1].astype(int)*self.gridSize])
                #print('No border')
                #exit()
'''