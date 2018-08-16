#
# This module contains the waterdrop erosion algorithm
#
import numpy as np


class WaterDrop():
    '''

    # -------------------TO DO-----------------------
    When using the CalculateZ method the values other than z can be stored tp be used in the next move step, this would
    increase performance as the same values do not need to be calculated twice.

    Look into if the erosionAmount calculation in the Erode() method needs to be altered.
    '''


    adjacentTilesTemplate = []
    erosionWeightTemplate = []
    erosionRowTemplate = [] # Not used
    erosionColumnTemplate = [] # Not used
    erosionTileTemplate = []

    heightMap = []


    def __init__(self, gridSize, x = None, y = None, inertia = 0.3, gravity = 10, evaporationRate = 0.02,
                 capacityMultiplier = 40, minimumSlope = 0.05, erosionRadius = 4, depositionRate = 0.1,
                 erosionRate = 0.1, numberOfSteps = 64, storeTrail = False):
        '''
        :param gridSize:
        :param x:
        :param y:
        :param inertia: Determines how easily the drop changes direction.
        :param gravity: Sets the maximum velocity
        :param evaporationRate:
        :param capacityMultiplier:
        :param minimumSlope: Prevents the sediment capacity from becoming negative.
        :param erosionradius:
        :param depositionRate:
        :param erosionRate:
        :param numberOfSteps:
        :param storeTrail: A boolean value indicating if the trail is to be stored or not.
        '''
        self.gridSize = gridSize


        # If x and y values are not given they will be randomized to be within the grid.
        if x is None:
            self.x = gridSize * np.random.rand()
        else:
            self.x = x
        if y is None:
            self.y = gridSize * np.random.rand()
        else:
            self.y = y


        self.xWholePart = np.floor(self.x)
        self.yWholePart = np.floor(self.y)
        self.xDecimalPart = self.x - self.xWholePart
        self.yDecimalPart = self.y - self.yWholePart
        self.adjacentTiles = np.zeros((4, 2))
        self.adjacentTiles[:, 0] = self._periodic(self.adjacentTilesTemplate[:, 0] + self.yWholePart)
        self.adjacentTiles[:, 1] = self._periodic(self.adjacentTilesTemplate[:, 1] + self.xWholePart)
        self.adjacentHeights = self.heightMap[self.adjacentTiles[:, 0].astype(int), self.adjacentTiles[:, 1].astype(int)]


        # Interpolates the z coordinate value of the drop based on the surrounding heights.
        self.z = (1-self.xDecimalPart) * (self.adjacentHeights[0]*(1-self.yDecimalPart) + self.adjacentHeights[2]*self.yDecimalPart) \
            + self.xDecimalPart * (self.adjacentHeights[1] * (1-self.yDecimalPart) + self.adjacentHeights[3]*self.yDecimalPart)


        self.xPreviousStep = None
        self.yPreviousStep = None
        self.zPreviousStep = None

        self.storeTrail = storeTrail
        if self.storeTrail:
            self.trailData = np.zeros((3, numberOfSteps))
            self.step = 0 # The index used to access the trailData array.

        self.direction = np.array([0,0])
        self.numberOfUnimprovedSteps = 0
        self.heightDifference = None


        self.velocity = 0
        self.waterAmount = 1
        self.sedimentCapacity = 0
        self.sedimentAmount = 0


        self.inertia = inertia
        self.gravity = gravity
        self.capacityMultiplier = capacityMultiplier
        self.minimumSlope = minimumSlope
        self.erosionRadius = erosionRadius
        self.erosionRate = erosionRate


    def StoreState(self):
        '''
        The coordinates of the drop is stored. This is done in order for certain methods (eg. Erode()) to use the
        previous coordinates. It is also done in order for complete trail data to be stored for visualization.
        :return:
        '''
        # The coordinates are stored for later use.
        self.xPreviousStep = self.x
        self.yPreviousStep = self.y
        self.zPreviousStep = self.z


        # Stores coordinates in array in order to visualize.
        if self.storeTrail:
            # x-, y- and z-coordinates are stored.
            self.trailData[0, self.step] = self.xPreviousStep
            self.trailData[1, self.step] = self.yPreviousStep
            self.trailData[2, self.step] = self.zPreviousStep
            self.step += 1


    def Move(self):
        '''
        The drop is moved according to the gradient at the current coordinates. The gradient is calculated using the
        heightvalues of the 4 adjacent tiles.
        '''


        '''
        # Expand upon the CalculateZ() method. Values calculated in that method should be kept as to avoid
        # calculating the same thing twice.
        '''


        # The direction is updated
        self.UpdateDirection(self.xDecimalPart, self.yDecimalPart, self.adjacentHeights)


        # The drop is moved
        self.x += self.direction[0]
        self.y += self.direction[1]


        # The whole and decimal parts of the coordinates are updated
        self.xWholePart = np.floor(self.x)
        self.yWholePart = np.floor(self.y)
        self.xDecimalPart = self.x - self.xWholePart
        self.yDecimalPart = self.y - self.yWholePart


        # The adjacent tiles are selected using the adjacent tile template.
        self.adjacentTiles = np.zeros((4, 2))
        self.adjacentTiles[:, 0] = self._periodic(self.adjacentTilesTemplate[:, 0] + self.yWholePart)
        self.adjacentTiles[:, 1] = self._periodic(self.adjacentTilesTemplate[:, 1] + self.xWholePart)
        # Retrieves the map height for the adjacent tiles and interpolates the height of the water drop.
        self.adjacentHeights = self.heightMap[self.adjacentTiles[:, 0].astype(int), self.adjacentTiles[:, 1].astype(int)]


        # Interpolates the z coordinate value of the drop based on the surrounding heights.
        self.z = (1-self.xDecimalPart) * (self.adjacentHeights[0]*(1-self.yDecimalPart) + self.adjacentHeights[2]*self.yDecimalPart) \
            + self.xDecimalPart * (self.adjacentHeights[1] * (1-self.yDecimalPart) + self.adjacentHeights[3]*self.yDecimalPart)
        self.heightDifference = self.z - self.zPreviousStep



    def UpdateDirection(self, xDecimalPart, yDecimalPart, adjacentHeights):
        '''

        :param xDecimalPart:
        :param yDecimalPart:
        :param adjacentHeights:
        :return:
        '''
        # The gradient is computed. If necessary the gradient can be randomized in the case that it's a null vector.
        xGradient1 = adjacentHeights[1] - adjacentHeights[0]
        xGradient2 = adjacentHeights[3] - adjacentHeights[2]
        yGradient1 = adjacentHeights[2] - adjacentHeights[0]
        yGradient2 = adjacentHeights[3] - adjacentHeights[1]
        gradient = np.array([xGradient1*(1-yDecimalPart) + xGradient2*yDecimalPart, \
                             yGradient1 * (1 - xDecimalPart) + yGradient2 * xDecimalPart])
        gradient /= np.linalg.norm(gradient)


        # The direction of the water drop is updated. This should depend on the drops inertia (velocity and mass), right
        #  now a constant inertai is used.
        self.direction = self.direction * self.inertia - gradient * (1-self.inertia)
        self.direction /= np.linalg.norm(self.direction)


    def UpdateVelocity(self):
        '''
        The increase or decrease of the velocity is determined by the heightdifference which corresponds to the slope
        of the terrain. The maximum velocity possible is set to be the gravity value given in the constructor.

        :param self:
        :return:
        '''
        self.velocity = self.velocity/2 + np.arctan(np.abs(self.heightDifference)) * self.gravity/np.pi


    def DepositOrErode(self):
        '''
        The Method Updates the amount of sediment the drop can carry. Based on this amount and wether or not the drop
        has travelled uphill it is determined if the drop should erode or deposit. In the case of uphill travel all
        sediment if possible is deposited. If the amount of sediment is less than the capacity the drop will erode,
        otherwise it will deposit.
        :return:
        '''

        if self.heightDifference > 0:
            self.Deposit(depositAll = True)
        else:
            self.sedimentCapacity = np.max((-self.heightDifference,self.minimumSlope)) * self.velocity *\
                                    self.waterAmount * self.capacityMultiplier
            if self.sedimentAmount < self.sedimentCapacity:
                self.Erode()
            else:
                self.Deposit()


    def Deposit(self, depositAll = False):

        #depositionAmount

        if depositAll:
            #print('DEPOSIT ALL')
            pass
        else:
            #print('DEPOSIT')
            pass


    def Erode(self):
        '''

        :return:
        '''
        #print('ERODE')


        #
        # The Bug with deep holes should be fixed once water evaporation, deposition and drop termination has been implemented.
        #


        # Determines the amount of material to remove.
        erosionAmount = np.min(((self.sedimentCapacity-self.sedimentAmount)*self.erosionRate,-self.heightDifference))


        # Select tiles to erode using templates.
        xWholePart = np.floor(self.xPreviousStep)
        yWholePart = np.floor(self.yPreviousStep)

        erosionTiles = np.zeros((self.erosionTileTemplate[self.erosionRadius].shape[0], 2))
        erosionTiles[:, 0] = self._periodic(self.erosionTileTemplate[self.erosionRadius][:, 0] + yWholePart)
        erosionTiles[:, 1] = self._periodic(self.erosionTileTemplate[self.erosionRadius][:, 1] + xWholePart)


        # Material is removed from the map and sediment is added to the drop.
        self.heightMap[erosionTiles[:, 0].astype(int), erosionTiles[:, 1].astype(int)] -= erosionAmount*self.erosionWeightTemplate[self.erosionRadius]
        self.sedimentAmount += erosionAmount





    #
    # The following part of the class consist of utility methods like getters and setters.
    #
    def __call__(self):
        '''
        Called to simulate a single step.
        :return:
        '''
        self.StoreState()
        self.Move()
        self.UpdateVelocity()
        self.DepositOrErode()

    def __repr__(self):
        # This method is automatically used when an object of this
        # class is displayed using the print() function
        return("Drop is at x = {}, y = {}".format(self.x, self.y))

    def _periodic(self, value):
        return (value + self.gridSize) % self.gridSize

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = self._periodic(value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = self._periodic(value)

    @classmethod
    def LinkToHeightMap(cls, heightMap):
        # This is used in order for the drop to have access to the heightmap. If the heightMap were to be changed by
        # one drop all other drops would also notice the change.
        cls.heightMap = heightMap

    def LinkToTemplates(self, templateList):
        self.adjacentTilesTemplate = templateList[0]
        self.erosionWeightTemplate = templateList[1]
        self.erosionRowTemplate = templateList[2]
        self.erosionColumnTemplate = templateList[3]


    @classmethod
    def InitializeTemplates(cls, maximumErosionRadius):
        '''
        The method is used to create templates which are used by all drops during the simulation.
        :param maximumErosionRadius:
        :return:
        '''
        # This is used as a template every time adjacent tiles are to be selected.
        cls.adjacentTilesTemplate = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


        # Initializes lists used as templates when eroding multiple tiles.
        for radius in range(0, maximumErosionRadius):
            rows = range(-radius, radius, 1)
            columns = range(-radius, radius, 1)
            rowGrid, columnGrid = np.meshgrid(rows, columns)
            rowList = np.reshape(rowGrid, [rowGrid.size, 1])
            columnList = np.reshape(columnGrid, [columnGrid.size, 1])
            distances = np.sqrt(rowList ** 2 + columnList ** 2)
            rowList = rowList[distances < radius]
            columnList = columnList[distances < radius]


            # The template lists are expanded.
            cls.erosionWeightTemplate.append(
                (radius - distances[distances < radius]) / np.sum(radius - distances[distances < radius]))


            erosionTiles = np.zeros((cls.erosionWeightTemplate[radius].shape[0], 2))
            erosionTiles[:, 0] = rowList
            erosionTiles[:, 1] = columnList
            cls.erosionTileTemplate.append(erosionTiles)

            #cls.erosionRowTemplate.append(rowList[distances < radius])
            #cls.erosionColumnTemplate.append(columnList[distances < radius])



#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
''' 
   def Move(self):
        # The whole and decimal parts of the coordinates are extracted.
        xWholePart = np.floor(self.x)
        yWholePart = np.floor(self.y)
        xDecimalPart = self.x - xWholePart       # Should be left
        yDecimalPart = self.y - yWholePart       # Should be left


        # The adjacent tiles are selected using the adjacent tile template.
        adjacentTiles = np.zeros((4, 2))
        adjacentTiles[:, 0] = self._periodic(self.adjacentTilesTemplate[:, 0] + yWholePart)
        adjacentTiles[:, 1] = self._periodic(self.adjacentTilesTemplate[:, 1] + xWholePart)


        # Retrieves the map height for the adjacent tiles and interpolates the height of the water drop.
        adjacentHeights = self.heightMap[adjacentTiles[:, 0].astype(int), adjacentTiles[:, 1].astype(int)]


        # The gradient is computed. If necessary the gradient can be randomized in the case that it's a null vector.
        xGradient1 = adjacentHeights[1] - adjacentHeights[0]
        xGradient2 = adjacentHeights[3] - adjacentHeights[2]
        yGradient1 = adjacentHeights[2] - adjacentHeights[0]
        yGradient2 = adjacentHeights[3] - adjacentHeights[1]
        gradient = np.array([xGradient1*(1-yDecimalPart) + xGradient2*yDecimalPart, \
                             yGradient1 * (1 - xDecimalPart) + yGradient2 * xDecimalPart])
        gradient /= np.linalg.norm(gradient)


        # The direction of the water drop is updated. This should depend on the drops inertia (velocity and mass), right
        #  now a constant inertai is used.
        self.direction = self.direction * self.inertia - gradient * (1-self.inertia)
        self.direction /= np.linalg.norm(self.direction)


        # The z-coordinate before the movement is calculated and all coordinates are stored.
        self.z = self.CalculateZ(xDecimalPart, yDecimalPart, adjacentHeights)
        self.xPreviousStep = self.x
        self.yPreviousStep = self.y
        self.zPreviousStep = self.z


        self.xPreviousWholePart = self.xWholePart
        self.ypreviousWholePart = self.yWholePart



        # Stores coordinates in array in order to visualize.
        if self.storeTrail:
            # x-, y- and z-coordinates are stored.
            self.trailData[0, self.step] = self.x
            self.trailData[1, self.step] = self.y
            self.trailData[2, self.step] = self.z
            self.step += 1


        # The drop is moved
        self.x += self.direction[0]
        self.y += self.direction[1]
        self.z = self.CalculateZ()
        self.heightDifference = self.z - self.zPreviousStep
'''


