
import numpy as np
import scipy.spatial
#import cPickle
import pickle


def GetIcoSphere(numberOfDivisions):
    # The function checks if the icosphere exists on file, if not a new icosphere is created and stored to file.
    # The template is stored in a .pkl file.


    import Root_Directory
    fileName = Root_Directory.Path() + '/Templates/IcoSphere_' + str(numberOfDivisions) + '.pkl'
    print(fileName)

    try:
        # Attempts to read the world from file.
        import gc
        # gc : Garbage Collector.
        # Disabling the garbage collector when loading the .pkl file improves the load time slightly.
        gc.disable()
        fileToRead = open(fileName, 'rb')
        world = pickle.load(fileToRead)
        fileToRead.close()
        gc.enable()
        print('file was opened')
    except:
        # Creates a new world and stores it to file.
        world = IcoSphere(numberOfDivisions)
        fileToOpen = open(fileName, 'wb')
        pickle.dump(world, fileToOpen, pickle.HIGHEST_PROTOCOL)
        fileToOpen.close()
        print('new world was created')
    return world

class IcoSphereSimple():
    def __init__(self,
                 numberOfDivisions = 0):
        # A dictionary used to prevent duplicate vertices.
        self.middle_point_cache = {}

        # Golden ratio
        PHI = (1 + np.sqrt(5)) / 2
        # vertices = [x, y, z]
        self.vertices = [
            self._NewVertex(-1, PHI, 0),
            self._NewVertex(1, PHI, 0),
            self._NewVertex(-1, -PHI, 0),
            self._NewVertex(1, -PHI, 0),
            self._NewVertex(0, -1, PHI),
            self._NewVertex(0, 1, PHI),
            self._NewVertex(0, -1, -PHI),
            self._NewVertex(0, 1, -PHI),
            self._NewVertex(PHI, 0, -1),
            self._NewVertex(PHI, 0, 1),
            self._NewVertex(-PHI, 0, -1),
            self._NewVertex(-PHI, 0, 1),
        ]
        self.faces = [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        for i in range(numberOfDivisions):
            faces_subdiv = []
            for tri in self.faces:
                v1 = self._GetMiddlePoint(tri[0], tri[1])
                v2 = self._GetMiddlePoint(tri[1], tri[2])
                v3 = self._GetMiddlePoint(tri[2], tri[0])

                faces_subdiv.append([tri[0], v1, v3])
                faces_subdiv.append([tri[1], v2, v1])
                faces_subdiv.append([tri[2], v3, v2])
                faces_subdiv.append([v1, v2, v3])
            self.faces = faces_subdiv

        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)
        self.numberOfvertices = np.size(self.vertices, axis=0)
        self.radius = np.ones((self.numberOfvertices, 1))

    def _NewVertex(self, x, y, z):
        '''
        Returns the vertex coordinates after being scaled to a unit sphere.
        '''

        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i / radius for i in (x, y, z)]


    def _GetMiddlePoint(self, point1, point2):
        '''
        Finds the middle point between two points. If the new point do not already exist in the list of vertices it is
        added. The index of the vertix is returned from the method.
        '''

        # We check if we have already cut this edge first
        # to avoid duplicated verts
        smaller_index = min(point1, point2)
        greater_index = max(point1, point2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in self.middle_point_cache:
            return self.middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = self.vertices[point1]
        vert_2 = self.vertices[point2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2)]

        self.vertices.append(self._NewVertex(*middle))

        index = len(self.vertices) - 1
        self.middle_point_cache[key] = index

        return index


class IcoSphere(IcoSphereSimple):
    '''
    The icosphere is made by dividing an icosahedron.
    The initial values of the vertices and faces correspond to a regular icosahedron.
    '''

    #from memory_profiler import profile
    #@profile
    def __init__(self,
                 numberOfDivisions = 0):
        super().__init__(numberOfDivisions = numberOfDivisions)

        self.shortestDistance = np.linalg.norm(self.vertices[self.faces[0][0]] - self.vertices[self.faces[0][1]])
        self.neighbours = Neighbours(self.vertices,
                                     self.shortestDistance,
                                     maxNeighbourDistance = 10)

class Neighbours:
    def __init__(self,
                 vertices,
                 shortestDistance,
                 maxNeighbourDistance = 5):
        #
        # maxNeighbourDistance: The maximum distance used to calculate neighbours, measured in units of the shortest neighbour distance.
        #
        vertexKDTree = scipy.spatial.cKDTree(vertices)


        #Consider using tuples instaed of lists, this may reduce memory usage.


        '''
        self.IDList = ()
        self.distanceList = []

        for iVertice, vertice in enumerate(vertices):
            IDListTemp = ()
            distanceListTemp = []

            if False:
                for radius in range(maxNeighbourDistance + 1):
                    # 0.5 is used as a buffer.
                    neighbourIDs = vertexKDTree.query_ball_point(vertice, (radius + 0.5) * shortestDistance)
                    differences = vertice - vertices[neighbourIDs, :]
                    distances = np.linalg.norm(differences, axis=1)
                    distances = np.reshape(distances, (np.size(distances), 1))

                    IDListTemp.append(neighbourIDs)
                    distanceListTemp.append(distances)
            else:
                radius = maxNeighbourDistance

                neighbourIDs = vertexKDTree.query_ball_point(vertice, (radius + 0.5) * shortestDistance)
                differences = vertice - vertices[neighbourIDs, :]
                distances = np.linalg.norm(differences, axis=1)
                distances = np.reshape(distances, (np.size(distances), 1))

                #print(type(neighbourIDs))
                #IDListTemp.append(neighbourIDs)
                distanceListTemp.append(distances)
            self.IDList = self.IDList + (tuple(neighbourIDs),)
            self.distanceList.append(distanceListTemp)


        '''



        self.IDList = []
        self.distanceList = []

        for iVertice, vertice in enumerate(vertices):
            IDListTemp = []
            distanceListTemp = []

            if False:
                for radius in range(maxNeighbourDistance + 1):
                    # 0.5 is used as a buffer.
                    neighbourIDs = vertexKDTree.query_ball_point(vertice, (radius + 0.5) * shortestDistance)
                    differences = vertice - vertices[neighbourIDs, :]
                    distances = np.linalg.norm(differences, axis=1)
                    distances = np.reshape(distances, (np.size(distances), 1))

                    IDListTemp.append(neighbourIDs)
                    distanceListTemp.append(distances)
            else:
                for radius in [1, maxNeighbourDistance]:
                    #radius = maxNeighbourDistance

                    neighbourIDs = vertexKDTree.query_ball_point(vertice, (radius + 0.5) * shortestDistance)
                    differences = vertice - vertices[neighbourIDs, :]
                    distances = np.linalg.norm(differences, axis=1)
                    distances = np.reshape(distances, (np.size(distances), 1))

                    IDListTemp.append(neighbourIDs)
                    distanceListTemp.append(distances)
            self.IDList.append(IDListTemp)
            self.distanceList.append(distanceListTemp)







