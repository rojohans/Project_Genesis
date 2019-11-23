import numpy as np
from scipy import spatial

class Plate():
    numberOfPlatesCreated = 0 # Used for assigning a unique ID to each plate.

    def __init__(self,
                 vertices = None,
                 flowVectors = None):
        # self.vertices is a list of the [x, y, z] coordinates of each point belonging to the plate.
        # self.centerPoint is a [x, y, z] point corresponding to the center of the plate. Rotations will be done about
        #                  this point and this point is used for comparison with other plates.
        if vertices is None:
            self.vertices = []
            self.centerPoint = []
            self.numberOfvertices = 0
        else:
            self.vertices = vertices
            self.numberOfvertices = np.size(vertices, 0)
            self.centerPoint = np.mean(self.vertices, axis = 0)
            r = np.sqrt(self.centerPoint[0]**2 + self.centerPoint[1]**2 + self.centerPoint[2]**2)
            self.centerPoint /= r


        #self.thickness =

        #self.velocity
        #self.rotation = 0 # An angle [0, 2*pi]


        if flowVectors is None:
            self.averageFlowVector = []
        else:
            pass
            #self.averageFlowVector =

        self.ID = Plate.numberOfPlatesCreated
        Plate.numberOfPlatesCreated += 1

        self.adjacentPlateIDs = []






    def AddVertice(self):
        pass

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

    def UpdateAverage(self):
        self.centerPoint = np.mean(self.vertices, axis=0)
        r = np.sqrt(self.centerPoint[0] ** 2 + self.centerPoint[1] ** 2 + self.centerPoint[2] ** 2)
        self.centerPoint /= r

        tmp = np.array([0, 0, 0])
        #for vertex in self.vertices:
            #closestVertexID = self.worldKDTree.query(vertex)
            #tmp = self.worldFlowVectors[closestVertexID, :]



        #self.averageFlow = []
        pass

    @classmethod
    def UpdateKDTree(cls):
        #
        # cls.verticeIDs is a list of the plate ID corresponding with each vertex.
        #
        for iPlate, plate in enumerate(cls.plateList):
            if iPlate == 0:
                plateVertices = plate.vertices
                cls.verticeIDs = plate.ID * np.ones((plate.numberOfvertices, 1))
            else:
                plateVertices = np.append(plateVertices, plate.vertices, axis = 0)
                cls.verticeIDs = np.append(cls.verticeIDs, plate.ID*np.ones((plate.numberOfvertices, 1)), axis = 0)
        cls.verticeTree = spatial.cKDTree(plateVertices)

    @classmethod
    def LinkLists(cls,
                  plateList,
                  worldKDTree,
                  worldFlowVectors):
        #
        # Gives all the plates access to the list of plates.
        #
        #
        # One could give the plates access to the entire world. (Not just the tree and flow vectors)
        #
        cls.plateList = plateList
        cls.worldKDTree = worldKDTree
        cls.worldFlowVectors = worldFlowVectors



class PlateCollection():
    # Used for storing a list of plates to file.
    def __init__(self,
                 plateList = None):
        if plateList == None:
            self.plateList = []
            self.numberOfPlates = 0
        else:
            self.plateList = plateList
            self.numberOfPlates = np.size(plateList)

    def AddPlate(self, plate):
        self.plateList.append(plate)
        self.numberOfPlates += 1

    def RemovePlate(self):
        #self.numberOfPlates -= 1
        pass

    def MergePlates(self):
        # This method should compare each plates with it's adjacent plates and possibly merge with one of them.
        pass

    def SplitPlates(self):
        pass



















