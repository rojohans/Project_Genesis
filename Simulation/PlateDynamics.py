import numpy as np
from scipy import spatial
import Utility

class Plate():
    numberOfPlatesCreated = 0 # Used for assigning a unique ID to each plate.

    def __init__(self,
                 vertices = None,
                 thickness = 0.3):
        # self.vertices is a list of the [x, y, z] coordinates of each point belonging to the plate.
        # self.centerPoint is a [x, y, z] point corresponding to the center of the plate. Rotations will be done about
        #                  this point and this point is used for comparison with other plates.
        # self.thickness is a value between 0 and 2. The value determines how the plate is formed. For 0 only vertices
        #                of exactly the same flow vector can form a plate. For 2 all vertices can be added to the plate,
        #                independent of the flow vector.
        if vertices is None:
            self.vertices = []
            self.centerPoint = []
            self.numberOfVertices = 0
        else:
            self.vertices = vertices
            self.numberOfVertices = np.size(vertices, 0)
            self.centerPoint = np.mean(self.vertices, axis = 0)
            r = np.sqrt(self.centerPoint[0]**2 + self.centerPoint[1]**2 + self.centerPoint[2]**2)
            self.centerPoint /= r


        self.thickness = thickness

        #self.velocity
        #self.rotation = 0 # An angle [0, 2*pi]

        self.averageFlowVector = []


        self.ID = Plate.numberOfPlatesCreated
        Plate.numberOfPlatesCreated += 1

        self.adjacentPlateIDs = []




    def AddVertices(self, newVertices):
        self.vertices = np.append(self.vertices, newVertices, axis = 0)
        self.numberOfVertices += np.size(newVertices, 0)

        # Adds the interpolated flow to the new points.
        for vertex in newVertices:
            queryResult = self.worldKDTree.query(vertex)
            vertexFlow = self.worldFlowVectors[queryResult[1]:queryResult[1]+1, :]
            self.verticesFlow = np.append(self.verticesFlow, vertexFlow, axis = 0)


    def RemoveVertice(self):
        pass

    def CalculateAdjacentPlates(self):
        #
        # This method should find the plates adjacent to this plate and store their ID's.
        #
        # self.adjacentPlateIDs is a list of the ID of all the adjacent plates.
        #
        self.adjacentPlateIDs = []
        # One could store which vertices have neighbours in different plates, this should make the next function call cheaper.

        queryResult = self.verticeTree.query(self.vertices, k = 6)
        adjacentVerticesList = queryResult[1]

        for adjacentVertices in adjacentVerticesList: # Loops over the vertices in the plate.
            adjacentIDs = self.verticeIDs[adjacentVertices]
            for adjacentID in adjacentIDs: # Loops over the adjacent vertices
                if adjacentID[0] not in self.adjacentPlateIDs and adjacentID[0] != self.ID:
                    self.adjacentPlateIDs.append(int(adjacentID[0]))
        self.adjacentPlateIDs.sort()

    def UpdateFlow(self):
        for iVertex, vertex in enumerate(self.vertices):

            #phi, theta, radius = Utility.CaartesianToSpherical(vertex)
            # Phi unit vector
            #phiVector = np.array([-np.sin(phi), np.cos(phi), 0])
            # Theta unit vector
            #thetaVector = np.array([ -np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta), np.cos(theta)])



            queryResult = self.worldKDTree.query(vertex)
            vertexFlow = self.worldFlowVectors[queryResult[1]:queryResult[1]+1, :]

            #vertexFlow = np.array([np.dot(vertexFlow, thetaVector)[0], np.dot(vertexFlow, phiVector)[0], 0])
            #vertexFlow = np.reshape(vertexFlow, (1, 3))

            if iVertex == 0:
                self.verticesFlow = vertexFlow
            else:
                self.verticesFlow = np.append(self.verticesFlow, vertexFlow, axis = 0)
        #print(self.verticesFlow)

    def UpdateAverage(self):
        #
        # Updates the center point and the average flow vector.
        #

        self.centerPoint = np.array([0, 0, 0])
        for iVertex, vertex in enumerate(self.vertices):
            self.centerPoint = (self.centerPoint * iVertex + vertex) / (iVertex + 1)
            self.centerPoint /= np.sqrt(self.centerPoint[0] ** 2 + self.centerPoint[1] ** 2 + self.centerPoint[2] ** 2)
        #self.centerPoint = np.mean(self.vertices, axis=0)
        #r = np.sqrt(self.centerPoint[0] ** 2 + self.centerPoint[1] ** 2 + self.centerPoint[2] ** 2)
        #self.centerPoint /= r

        self.averageFlowVector = np.array([0, 0, 0], dtype = 'float64')

        for iVertex, vertex in enumerate(self.vertices):
        #for iVertex in range(self.numberOfVertices):
            #queryResult = self.worldKDTree.query(vertex)
            #vertexFlow = self.worldFlowVectors[queryResult[1], :]
            #if iVertex == 0:
            #    self.flowVectors = vertexFlow
            #else:
            #    self.flowVectors = np.append(self.flowVectors, vertexFlow, axis = 0)

            #vertex = self.vertices[iVertex, :]
            vertexFlow = self.verticesFlow[iVertex, :]

            #vertexFlowAtCenter = Utility.RotateVector(vertex, self.centerPoint, vertexFlow) ------- not done when rotation axises are used.
            #self.averageFlowVector += vertexFlowAtCenter

            #elf.averageFlowVector = (self.averageFlowVector * iVertex + vertexFlowAtCenter) / (iVertex+1)
            #self.averageFlowVector = (self.averageFlowVector * iVertex + vertexFlow) / (iVertex + 1)
            self.averageFlowVector += vertexFlow
            #self.averageFlowVector /= np.sqrt(self.averageFlowVector[0]**2 + self.averageFlowVector[1]**2 + self.averageFlowVector[2]**2)

            #self.averageFlowVector += vertexFlow
        self.averageFlowVector /= self.numberOfVertices
        #self.averageFlowVector /= Utility.VectorDistance(self.averageFlowVector, np.array([0, 0, 0]))
        #self.averageFlowVector /= np.sqrt( self.averageFlowVector[0]**2 + self.averageFlowVector[1]**2 + self.averageFlowVector[2]**2 )
        #print('------------------')

    def CalculateStress(self):
        self.stressVector = np.zeros((self.numberOfVertices, 1))
        for iVertex, vertex in enumerate(self.vertices):
            vectorAtCenter = self.verticesFlow[iVertex, :]
            #vectorAtCenter = Utility.RotateVector(vertex, self.centerPoint, self.verticesFlow[iVertex, :]) ------------ not done when rotation axises are used.
            self.stressVector[iVertex] = Utility.VectorDistance(self.averageFlowVector, vectorAtCenter)

    @classmethod
    def UpdateKDTree(cls):
        #
        # cls.verticeIDs is a list of the plate ID corresponding with each vertex.
        #
        #for iPlate, plate in enumerate(cls.plateList):
        iPlate = -1
        for key, plate in cls.plateDictionary.items():
            iPlate += 1
            if iPlate == 0:
                plateVertices = plate.vertices
                cls.verticeIDs = plate.ID * np.ones((plate.numberOfVertices, 1))
            else:
                plateVertices = np.append(plateVertices, plate.vertices, axis = 0)
                cls.verticeIDs = np.append(cls.verticeIDs, plate.ID*np.ones((plate.numberOfVertices, 1)), axis = 0)
        cls.verticeTree = spatial.cKDTree(plateVertices)

    @classmethod
    def LinkLists(cls,
                  plateDictionary,
                  worldKDTree,
                  worldFlowVectors,
                  world):
        #
        # Gives all the plates access to the list of plates.
        #
        #
        # One could give the plates access to the entire world. (Not just the tree and flow vectors)
        #
        cls.plateDictionary = plateDictionary
        cls.worldKDTree = worldKDTree
        cls.worldFlowVectors = worldFlowVectors
        cls.world = world

    @classmethod
    def Initialize(cls,
                   minimumPlateSize):
        #
        # If a plate is smaller than the minimumPlateSize, then it should be merged with an adjacent plate.
        #
        cls.minimumPlateSize = minimumPlateSize

    @classmethod
    def CheckForMerge(cls):
        #
        # Loops over all plates and checks if adjacent plate should be merged into new plates.
        #

        keyList = [key for key in cls.plateDictionary]
        visitedKeys = []

        while np.size(keyList) != 0:
            key = np.random.choice(keyList)
            plate = cls.plateDictionary[key]
            adjacentIDs = plate.adjacentPlateIDs.copy()
            #print(key)
            #print(adjacentIDs)
            #print('-----------')
            for iAdjacent, adjacentID in enumerate(adjacentIDs):
                if adjacentID in visitedKeys:
                    continue
                adjacentPlate = cls.plateDictionary[adjacentID]

                combinedPlate = Plate(vertices=np.append(plate.vertices, adjacentPlate.vertices, axis=0),
                                      thickness=plate.thickness)

                combinedPlate.UpdateFlow()
                combinedPlate.UpdateAverage()
                combinedPlate.CalculateStress()

                #print(np.max(combinedPlate.stressVector))
                if np.max(combinedPlate.stressVector) < plate.thickness:
                    #print(key)
                    #print(adjacentID)
                    #print('----------')
                    cls.MergePlates(plate, adjacentPlate)
                    if adjacentID in keyList:
                        keyList.remove(adjacentID)
                        visitedKeys.append(adjacentID)
                    plate.UpdateFlow()
                    plate.UpdateAverage()

            keyList.remove(key)
            visitedKeys.append(key)



        '''
        potentialCombinationsKeys = []
        potentialCombinationStress = []

        for key in keyList:
            plate = cls.plateDictionary[key]
            #print(plate.ID)
            #print(plate.adjacentPlateIDs)
            tmp1 = [] # Stores IDs
            tmp2 = [] # Stores maximum stress
            for iAdjacent, adjacentID in enumerate(plate.adjacentPlateIDs):
                adjacentPlate = cls.plateDictionary[adjacentID]


                combinedPlate = Plate(vertices = np.append(plate.vertices, adjacentPlate.vertices, axis = 0),
                                      thickness = plate.thickness)

                combinedPlate.UpdateFlow()
                combinedPlate.UpdateAverage()
                combinedPlate.CalculateStress()

                if np.max(combinedPlate.stressVector) < plate.thickness:
                    tmp1.append(adjacentID)
                    tmp2.append(np.max(combinedPlate.stressVector))

            potentialCombinationsKeys.append(tmp1)
            potentialCombinationStress.append(tmp2)

        #for key in keyList:
        #    print(potentialCombinationsKeys[key])
        #    print(potentialCombinationStress[key])
        #    print('----------------')


        platesJustMerged = True
        while platesJustMerged is True:
            platesJustMerged = False

            lowestStress = 1000
            lowestStressKey = None
            for key in keyList:
                if key in visitedKeys:
                    continue
                if np.size(potentialCombinationStress[key]) == 0:
                    continue
                try:
                    if np.min(potentialCombinationStress[key]) < lowestStress:
                        lowestStress = np.min(potentialCombinationStress[key])
                        lowestStressKey = key
                except:
                    print('--- ERROR ---')
                    print(np.size(potentialCombinationStress[key]))
                    print(potentialCombinationStress[key])
                    print('--- ERROR ---')
                    quit()
            if lowestStressKey is not None:
                platesJustMerged = True
                tmp = 100
                TMP = 0
                for i, potentialKey in enumerate(potentialCombinationsKeys[lowestStressKey]):
                    if potentialCombinationStress[lowestStressKey][i] < tmp:
                        tmp = potentialCombinationStress[lowestStressKey][i]
                        TMP = potentialKey

                mainPlate = cls.plateDictionary[lowestStressKey]
                try:
                    secondaryPlate = cls.plateDictionary[TMP]
                except:
                    print('--- ERROR TMP---')
                    print(TMP)
                    print('--- ERROR TMP ---')
                    quit()

                try:
                    cls.MergePlates(mainPlate, secondaryPlate)
                except:
                    print('--- ERROR plate marge ---')
                    print(mainPlate)
                    print(mainPlate.ID)
                    print('--- ERROR plate marge ---')
                    quit()

                #print(lowestStressKey)
                #print(potentialCombinationsKeys[lowestStressKey])
                #print(potentialCombinationStress[lowestStressKey])


                visitedKeys.append(lowestStressKey)
                visitedKeys.append(TMP)

                for key in keyList:
                    for i, non in enumerate(potentialCombinationsKeys[key]):
                        if potentialCombinationsKeys[key][i] == lowestStressKey or potentialCombinationsKeys[key][i] == TMP:
                            del potentialCombinationsKeys[key][i]
                            del potentialCombinationStress[key][i]
        quit()
        '''


        if False:
            platesJustMerged = False # don't think it's use.

            plateList = [plate for key, plate in cls.plateDictionary.items()]
            keyList = [key for key in cls.plateDictionary]
            visitedKeys = []

            for key in keyList:
            #for key, plate in cls.plateDictionary.items():
            #for plate in plateList:
                if key not in visitedKeys:
                    plate = cls.plateDictionary[key]
                    #print('--------------------------------')
                    #print(plate.ID)
                    #print(plate.adjacentPlateIDs)
                    for adjacentID in plate.adjacentPlateIDs.copy():
                        if adjacentID not in visitedKeys:
                            #print(adjacentID)
                            #print('-------------------------')
                            #print(plate.ID)
                            #print(adjacentID)


                            #try:
                            adjacentPlate = cls.plateDictionary[adjacentID] # error ?????

                            #tmp = Utility.RotateVector2Steps(adjacentPlate.centerPoint.copy(),
                            #                                 plate.centerPoint.copy(),
                            #                                 adjacentPlate.averageFlowVector.copy())
                            tmp = Utility.RotateVector(adjacentPlate.centerPoint.copy(),
                                                       plate.centerPoint.copy(),
                                                       adjacentPlate.averageFlowVector.copy())
                            # print('------')
                            flowDifference = Utility.VectorDistance(plate.averageFlowVector, tmp)
                            #flowDifference = Utility.VectorDistance(plate.averageFlowVector, adjacentPlate.averageFlowVector)

                            averageThickness = (plate.thickness * plate.numberOfVertices +
                                                adjacentPlate.thickness * adjacentPlate.numberOfVertices) / \
                                               (plate.numberOfVertices + adjacentPlate.numberOfVertices)
                            if flowDifference <= averageThickness:
                                bothPlatesFlowVectors = np.append(plate.verticesFlow.copy(), adjacentPlate.verticesFlow.copy(), axis = 0)
                                bothPlatesVertices = np.append(plate.vertices.copy(), adjacentPlate.vertices.copy(), axis=0)

                                #flowMean = np.mean(bothPlatesFlowVectors, axis = 0)
                                #flowMean /= np.sqrt(flowMean[0]**2 + flowMean[1]**2 + flowMean[2]**2)

                                #centerPoint = np.mean(bothPlatesVertices, axis=0)
                                centerPoint = (plate.centerPoint*plate.numberOfVertices +
                                               adjacentPlate.centerPoint+adjacentPlate.numberOfVertices) /\
                                              (plate.numberOfVertices + adjacentPlate.numberOfVertices)

                                r = np.sqrt(centerPoint[0] ** 2 + centerPoint[1] ** 2 + centerPoint[2] ** 2)
                                centerPoint /= r

                                # Calculates the mean flow at the center point for the combined plate.
                                flowMean = np.array([0, 0, 0])
                                for iVertex in range(plate.numberOfVertices + adjacentPlate.numberOfVertices):
                                    flowMean = (flowMean*iVertex + bothPlatesFlowVectors[iVertex, :]) / (iVertex + 1)


                                a = 5
                                for iVertex in range(plate.numberOfVertices + adjacentPlate.numberOfVertices):
                                    #tmp = Utility.RotateVector2Steps(bothPlatesVertices[iVertex, :].copy(),
                                    #                                 centerPoint.copy(),
                                    #                                 bothPlatesFlowVectors[iVertex, :].copy())
                                    tmp = Utility.RotateVector(bothPlatesVertices[iVertex, :].copy(),
                                                               centerPoint.copy(),
                                                               bothPlatesFlowVectors[iVertex, :].copy())
                                    b = Utility.VectorDistance(tmp, flowMean)
                                    #b = Utility.VectorDistance(bothPlatesFlowVectors[iVertex, :], flowMean)
                                    try:
                                        if b > averageThickness:
                                            #print(b)
                                            a -= 1
                                    except:
                                        print('|---> ERROR <---|')
                                        print(tmp)
                                        print(flowMean)
                                        print(np.shape(flowMean))
                                        print(b)
                                        print(averageThickness)
                                        print('|---> ERROR <---|')
                                        quit()





                                # print(plate.ID)
                                # print(plate.numberOfVertices)
                                # print(adjacentPlate.numberOfVertices)



                                #if flowDifference <= averageThickness:
                                if a > 0:
                                    # print('Plates should merge')
                                    # print(adjacentPlate.ID)
                                    visitedKeys.append(adjacentPlate.ID)
                                    cls.MergePlates(plate, adjacentPlate)
                                    plate.UpdateAverage()
                                    #platesJustMerged = True
                                #except:
                                #    print('---ERROR---')
                                #pass
                                #print(plate.adjacentPlateIDs)
                                #print(adjacentID)
                                #print('---ERROR---')
                                #quit()


                    visitedKeys.append(key)
                    #if platesJustMerged:
                    #    platesJustMerged = False
                    #    break

            '''
            keyList = [key for key in cls.plateDictionary]
            visitedKeys = []
            for key in keyList:
                if key not in visitedKeys:
                    plate = cls.plateDictionary[key]
                    if plate.numberOfVertices < cls.minimumPlateSize:
                        visitedKeys.append(key)
                        bestPlateSize = np.inf
                        bestPlateID = None
                        for adjacentID in plate.adjacentPlateIDs:
                            # The small plate should join the adjacent plate with the most similar flow vector, not the largest or smallest plate.
                            if cls.plateDictionary[adjacentID].numberOfVertices < bestPlateSize:
                                bestPlateID = adjacentID
                        #print(plate.ID)
                        #print(bestPlateID)
                        #print(plate.adjacentPlateIDs)
                        #print(cls.plateDictionary[bestPlateID].adjacentPlateIDs)
                        #print('------------------------------')
                        cls.MergePlates(cls.plateDictionary[bestPlateID], plate)
                        #print('Small plate merged into larger one.')
                        cls.plateDictionary[bestPlateID].UpdateAverage()
            '''

    @classmethod
    def MergePlates(cls,
                    plate1,
                    plate2):
        #
        # Adds plate2 to plate1.
        #

        plate1.AddVertices(plate2.vertices)

        # Loops over all the plates and makes changes to their adjacentPlateIDs lists.
        for key, plate in cls.plateDictionary.items():
            if key == plate1.ID:
                # removes the merged plate from the host plate and adds plates which the removed plate had as neighbours.
                #print('-------------')
                #print(plate.adjacentPlateIDs)

                #print(plate.adjacentPlateIDs)
                #print(plate2.adjacentPlateIDs)
                for element in plate2.adjacentPlateIDs:
                    plate.adjacentPlateIDs.append(element)
                #print(plate.adjacentPlateIDs)
                if plate1.ID in plate.adjacentPlateIDs:
                    plate.adjacentPlateIDs.remove(plate1.ID)
                if plate2.ID in plate.adjacentPlateIDs:
                    plate.adjacentPlateIDs.remove(plate2.ID)

                plate.adjacentPlateIDs = list(set(plate.adjacentPlateIDs))
                plate.adjacentPlateIDs.sort()
                #print(plate.adjacentPlateIDs)
                #quit()
            else:
                '''
                if plate2.ID in plate.adjacentPlateIDs:
                    # Changes the adjacency ID of plates adjacent to the now removed plate to the host plate ID instead.
                    for iID, ID in enumerate(plate.adjacentPlateIDs):
                        if ID == plate2.ID:
                            plate.adjacentPlateIDs[iID] = plate1.ID
                    plate.adjacentPlateIDs = list(set(plate.adjacentPlateIDs))
                    plate.adjacentPlateIDs.sort()
                '''
                if plate2.ID in cls.plateDictionary[key].adjacentPlateIDs:
                    # Changes the adjacency ID of plates adjacent to the now removed plate to the host plate ID instead.
                    for iID, ID in enumerate(cls.plateDictionary[key].adjacentPlateIDs):
                        if ID == plate2.ID:
                            cls.plateDictionary[key].adjacentPlateIDs[iID] = plate1.ID
                    cls.plateDictionary[key].adjacentPlateIDs = list(set(cls.plateDictionary[key].adjacentPlateIDs))
                    cls.plateDictionary[key].adjacentPlateIDs.sort()
        del cls.plateDictionary[plate2.ID]
        #plate1.UpdateAverage()



class PlateCollection():
    # Used for storing a list of plates to file.
    def __init__(self,
                 plateDictionary = None,
                 xFlow = None,
                 yFlow = None,
                 zFlow = None):
        if plateDictionary == None:
            self.plateDictionary = []
        else:
            self.plateList = plateDictionary

        self.xFlow = xFlow
        self.yFlow = yFlow
        self.zFlow = zFlow

    def AddPlate(self, plate):
        self.plateDictionary[plate.ID] = plate

    def RemovePlate(self):
        #self.numberOfPlates -= 1
        pass

    def MergePlates(self):
        # This method should compare each plates with it's adjacent plates and possibly merge with one of them.
        pass

    def SplitPlates(self):
        pass



















