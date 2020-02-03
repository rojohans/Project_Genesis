import numpy as np
from scipy import spatial
import Utility
import scipy.spatial

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

    def UpdateNearestPointIndex(self):
        #
        # This method should probably be included in the initializer or Plate.AddVertice()
        #
        for iVertex, vertex in enumerate(self.vertices):
            queryResult = self.worldKDTree.query(vertex, 2)
            #print(queryResult)
            #quit()

            if iVertex == 0:
                self.nearestPointIndex = [queryResult[1][0]]
                self.secondNearestPointIndex = [queryResult[1][1]]
            else:
                self.nearestPointIndex.append(queryResult[1][0])
                self.secondNearestPointIndex.append(queryResult[1][1])
        #print(self.nearestPointIndex)


    def UpdateFlow(self):
        if False:
            for iVertex, vertex in enumerate(self.vertices):
                #queryResult = self.worldKDTree.query(vertex, 1)
                queryResult = self.worldKDTree.query(vertex, 1, distance_upper_bound = 1.5*self.world.shortestDistance)
                vertexFlow = self.worldFlowVectors[queryResult[1]:queryResult[1]+1, :]

                if iVertex == 0:
                    self.verticesFlow = vertexFlow
                else:
                    self.verticesFlow = np.append(self.verticesFlow, vertexFlow, axis = 0)
        else:
            queryResult = self.world.kdTree.query(self.vertices)
            #print(queryResult[1])
            self.verticesFlow = self.worldFlowVectors[queryResult[1], :]
            #quit()

    def UpdateAverage(self):
        #
        # Updates the center point and the average flow vector.
        #

        '''
        self.centerPoint = np.array([0, 0, 0])
        for iVertex, vertex in enumerate(self.vertices):
            self.centerPoint = (self.centerPoint * iVertex + vertex) / (iVertex + 1)
            self.centerPoint /= np.sqrt(self.centerPoint[0] ** 2 + self.centerPoint[1] ** 2 + self.centerPoint[2] ** 2)
        '''
        self.centerPoint = np.mean(self.vertices, axis = 0)
        self.centerPoint /= Utility.VectorDistance(self.centerPoint, np.array([0, 0, 0]))

        #self.centerPoint = np.mean(self.vertices, axis=0)
        #r = np.sqrt(self.centerPoint[0] ** 2 + self.centerPoint[1] ** 2 + self.centerPoint[2] ** 2)
        #self.centerPoint /= r
        '''
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
        '''
        self.averageFlowVector = np.mean(self.verticesFlow, axis = 0)
        #self.averageFlowVector /= self.numberOfVertices
        #self.averageFlowVector /= Utility.VectorDistance(self.averageFlowVector, np.array([0, 0, 0]))
        #self.averageFlowVector /= np.sqrt( self.averageFlowVector[0]**2 + self.averageFlowVector[1]**2 + self.averageFlowVector[2]**2 )
        #print('------------------')

    def CalculateStress(self):
        self.stressVector = np.zeros((self.numberOfVertices, 1))
        for iVertex, vertex in enumerate(self.vertices):
            vectorAtCenter = self.verticesFlow[iVertex, :]
            #vectorAtCenter = Utility.RotateVector(vertex, self.centerPoint, self.verticesFlow[iVertex, :]) ------------ not done when rotation axises are used.
            self.stressVector[iVertex] = Utility.VectorDistance(self.averageFlowVector, vectorAtCenter)

    def CalculateInteractionStress(self):
        '''

        :return:
        '''


        #
        #             Consider using adjacentPlate.borderKDTree instead of surfaceKDTree
        #

        self.interactionStress = np.zeros((self.numberOfVertices, 1))
        tmpStress = np.zeros((self.numberOfVertices, 1))
        tmpCount = np.zeros((self.numberOfVertices, 1))
        for adjacentID in self.adjacentPlateIDs:
            adjacentPlate = self.plateDictionary[adjacentID]
            queryResult = adjacentPlate.surfaceKDTree.query_ball_point(self.vertices,
                                                                       r=7.2 * self.world.shortestDistance)
            #print(adjacentID)
            for iPoint, result in enumerate(queryResult):
                if result:
                    #interactionFlow = adjacentPlate.verticesFlow[result, :]
                    for iAdjacent in result:
                        diff = Utility.VectorDistance(self.verticesFlow[iPoint, :], adjacentPlate.verticesFlow[iAdjacent, :])

                        # This has been commented for the sake of performance, it makes the stress dependent on the
                        # angle between the interacting plates.
                        #crossVector = np.cross(self.vertices[iPoint, :], adjacentPlate.vertices[iAdjacent, :])
                        #dotResult = np.dot(self.verticesFlow[iPoint, :], crossVector)
                        #if dotResult < 0:
                        #    dotResult = 0
                        #diff *= dotResult


                        tmpStress[iPoint, 0] += diff
                        tmpCount[iPoint, 0] += 1
            '''
            queryResult = adjacentPlate.borderKDTree.query_ball_point(self.vertices,
                                                                       r=7.2 * self.world.shortestDistance)
            #print(adjacentID)

            for iPoint, result in enumerate(queryResult):
                if result:
                    try:
                        adjacentPlate.borderIndex = np.array(adjacentPlate.borderIndex)
                        #         (N, ) -> (N, 1)
                        borderResult = adjacentPlate.borderIndex[result, :]
                    except:
                        print(result)
                        print(adjacentPlate.verticesFlow[result, :])
                        print(np.shape(adjacentPlate.verticesFlow))
                        print(type(adjacentPlate.verticesFlow))
                        print(np.shape(adjacentPlate.borderIndex))
                        print(type(adjacentPlate.borderIndex))
                        quit()
                    #interactionFlow = adjacentPlate.verticesFlow[result, :]
                    for iAdjacent in borderResult:
                        diff = Utility.VectorDistance(self.verticesFlow[iPoint, :], adjacentPlate.verticesFlow[iAdjacent, :])
                        crossVector = np.cross(self.vertices[iPoint, :], adjacentPlate.vertices[iAdjacent, :])
                        dotResult = np.dot(self.verticesFlow[iPoint, :], crossVector)
                        if dotResult < 0:
                            dotResult = 0
                        diff *= dotResult


                        tmpStress[iPoint, 0] += diff
                        tmpCount[iPoint, 0] += 1
            '''
        #tmpStress[tmpCount > 0] /= tmpCount[tmpCount > 0]
        self.interactionStress = tmpStress / 2

    def Rotate(self,
               axisOfRotation = None,
               angleOfRotation = None,
               angleScaling = 1):
        avgVecNorm = None
        if axisOfRotation is None:
            avgVecNorm = Utility.VectorDistance(self.averageFlowVector, np.array([0, 0, 0]))
            axisOfRotation = self.averageFlowVector / avgVecNorm
        if angleOfRotation is None:
            if avgVecNorm is not None:
                angleOfRotation = angleScaling * avgVecNorm * np.pi/180
            else:
                angleOfRotation = angleScaling * Utility.VectorDistance(axisOfRotation, np.array([0, 0, 0])) * np.pi/180
        self.vertices = Utility.RotateAroundAxis(self.vertices, axisOfRotation, angleOfRotation)

    def NormalizeVertices(self):
        # This might be needed if the vertices drifts away from the unit sphere due to the rotation matrix.
        for iPoint in range(self.numberOfVertices):
            self.vertices[iPoint, :] /= Utility.VectorDistance(self.vertices[iPoint, :], np.array([0, 0, 0]))

    def MeshTriangulation(self, triangleSizeRegulation = 1.1):
        #
        # triangleSizeRegulation : A float value indicating the size of the largest triangle allowed. A value of 1
        # represents triangle of the size present in the underlying world mesh, these triangles were created from
        # polyhedron division. A value greater than 1 will allow bigger triangles than the "default ones."
        #
        if self.numberOfVertices >=4:
            tri = scipy.spatial.ConvexHull(self.vertices)
            plateFaces = tri.simplices

            facesToRemove = []
            for iFace, face in enumerate(plateFaces):
                faceCenter = (self.vertices[face[0], :] + self.vertices[face[1], :] + self.vertices[face[2], :]) / 3
                distanceToCenter1 = Utility.VectorDistance(self.vertices[face[0], :], faceCenter)
                distanceToCenter2 = Utility.VectorDistance(self.vertices[face[1], :], faceCenter)
                distanceToCenter3 = Utility.VectorDistance(self.vertices[face[2], :], faceCenter)
                if (distanceToCenter1 + distanceToCenter2 + distanceToCenter3) / 3 > 1.1 * 2 * self.world.shortestDistance / 3:
                    facesToRemove.append(iFace)
            plateFaces = np.delete(plateFaces, facesToRemove, axis=0)
            self.triangles = plateFaces
        else:
            self.triangles = []

    def FindBorderPoints(self):
        '''
        This method updates the self.borderVertex array.
        This is done by looping over self.triangles, vertices which are part of just one edge are considered to be on
        the border of the surface.

        |  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        |  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        | This method is extremely slow and could probably be sped up.
        |  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        |  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        :return:
        '''

        edgeList = []
        for triangle in self.triangles:
            edgeList.append([triangle[0], triangle[1]])
            edgeList.append([triangle[0], triangle[2]])
            edgeList.append([triangle[1], triangle[2]])

        def justonce(l):
            #
            # This method returns the elements of l which only occurs once.
            # The method is extremely slow.
            #
            once = []
            more = []
            for x in l:
                if x not in more:
                    if x in once:
                        more.append(x)
                        once.remove(x)
                    else:
                        once.append(x)
            return once
        once = justonce(edgeList)
        borderIndex = []
        for edge in once:
            borderIndex.extend(edge)
        self.borderIndex =  list(set(borderIndex))
        self.borderVertex = self.vertices[self.borderIndex, :]
        self.borderEdges = once
        #self.borderEdges = np.array(once)

    def ConnectBorderPoints(self, sort = False):
        '''
        This method sorts the edge points such that a line can be easily drawn through them, creating a border around the plate.
        A dictionary is created containing all the edges. A random edge is selected and the line segment is created by
        stepping from one point to another a along the edges to create a connected line.
        Id the surface is split into several surfaces (2-3 points i bit away from the main surface) those will not be
        included in the line, this is a problem. One rare problem is that a line is drawn only around those isolated
        points. One should probably make a loop such that all the border points are included in the line segment.
        :return:
        '''

        if sort:
            # A dictionary is created which indicates which points are connected to which.
            linkDictionary = {}
            for edgeIndex in self.borderIndex:
                linkDictionary[edgeIndex] = []
            for edge in self.borderEdges:
                linkDictionary[edge[0]].append(edge[1])
                linkDictionary[edge[1]].append(edge[0])

            # A random edge is selected. This edge could be part of a small isolated part of the surface, in that case the
            # visualization would not be ideal. One should probably repeat the whole procedure until all border points are
            # part of the line list.
            remainingBorderIndices = self.borderIndex.copy()
            visitedEdges = []
            visitedPoints = []

            # Adds line segments step by step by going through connected edges. The loop will end when the line returns to
            # the same point at which it started.
            while remainingBorderIndices:
                #print('------------')
                #print(remainingBorderIndices)
                #print(np.shape(remainingBorderIndices))

                randomPoint = [np.random.choice(remainingBorderIndices)]
                previousPoint = randomPoint[0]
                currentPoint = linkDictionary[previousPoint][0].copy()
                visitedEdges.append([previousPoint, currentPoint])
                visitedPoints.append(previousPoint)
                visitedPoints.append(currentPoint)
                remainingBorderIndices.remove(previousPoint)
                if currentPoint in remainingBorderIndices:
                    remainingBorderIndices.remove(currentPoint)

                startFound = False
                while startFound == False:
                    adjacentPoints = linkDictionary[currentPoint].copy()
                    adjacentPoints.remove(previousPoint)
                    while True:
                        nextPoint = adjacentPoints[0]
                        nextEdge = [currentPoint, nextPoint]

                        if nextEdge not in visitedEdges:
                            break
                        else:
                            adjacentPoints.remove(nextPoint)
                            if not adjacentPoints:
                                startFound = True
                                break


                    visitedPoints.append(nextPoint)
                    visitedEdges.append(nextEdge)
                    previousPoint = currentPoint
                    currentPoint = nextPoint
                    if nextPoint in remainingBorderIndices:
                        remainingBorderIndices.remove(nextPoint)
                #break

            #print(visitedPoints)
            #print('----')






            #a = np.arange(0, np.size(visitedPoints, 0))
            self.borderLines=np.array(visitedEdges)

            #print(type(self.borderLines))
            #quit()

            #a = np.array([1, 2, 2, 1]).reshape(2, 2)
            # palette must be given in sorted order
            #palette = [1, 2]
            palette = list(np.sort(visitedPoints))

            a = np.zeros((np.size(visitedPoints, 0), 1))
            for iPoint, point in enumerate(visitedPoints):
                for i, p in enumerate(palette):
                    if p == point:
                        a[i] = iPoint
                        #break

            #print(a)
            #print(palette)
            #print(np.shape(a))
            #print(np.shape(palette))

            #print(np.append(a, np.array(palette)))
            #print(palette)
            #quit()

            #print(palette)
            # key gives the new values you wish palette to be mapped to.
            #key = np.array([0, 10])
            #key = a
            index = np.digitize(self.borderLines.ravel(), palette, right=True)
            #print(a[index].reshape(self.borderLines.shape))

            c = self.borderLines.copy()
            self.borderLines = a[index].reshape(self.borderLines.shape)
            #print(self.borderLines)
            #print(np.append(c, self.borderLines, axis = 1))
            #quit()



            self.borderVertex = self.vertices[visitedPoints]
            #self.borderLines = np.vstack(
            #    [np.arange(0, np.size(self.borderVertex, 0) - 1.5),
            #     np.arange(1, np.size(self.borderVertex, 0) - .5)]
            #).T
            #print(visitedEdges)
            #print('=========================================================')
            #print('=========================================================')
            #print('=========================================================')
            return

        else:

            a = np.arange(0, np.size(self.borderIndex, 0))
            self.borderLines=np.array(self.borderEdges)

            #print(type(self.borderLines))
            #quit()

            #a = np.array([1, 2, 2, 1]).reshape(2, 2)
            # palette must be given in sorted order
            #palette = [1, 2]
            palette = list(np.sort(self.borderIndex))
            #print(palette)
            # key gives the new values you wish palette to be mapped to.
            #key = np.array([0, 10])
            #key = a
            index = np.digitize(self.borderLines.ravel(), palette, right=True)
            #print(a[index].reshape(self.borderLines.shape))

            c = self.borderLines.copy()
            self.borderLines = a[index].reshape(self.borderLines.shape)
            #print(np.append(c, self.borderLines, axis = 1))
            #quit()


            return
            #print(type(self.borderLines))
            #quit()


    def UpdateBorderKDTree(self):
        self.borderKDTree = spatial.cKDTree(self.borderVertex)
    def UpdateSurfaceKDTree(self):
        self.surfaceKDTree = spatial.cKDTree(self.vertices)

    def FindSecondBorderPoints(self):
        '''
        :return:
        '''

        a = []

        queriedIndices = []
        for borderVertex in self.borderVertex:
            queryResult = self.surfaceKDTree.query_ball_point(borderVertex, 1.2*self.world.shortestDistance)
            queriedIndices.extend(queryResult)
        queriedIndices = set(queriedIndices)
        #print(queriedIndices)
        #print(self.borderIndex)
        borderIndices = self.borderIndex.copy()

        secondBorderIndices = list(queriedIndices - set(borderIndices))
        #print(secondBorderIndices)
        self.secondBorderIndices = secondBorderIndices
        self.secondBorderPoints = self.vertices[secondBorderIndices, :]
        #quit()







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



















