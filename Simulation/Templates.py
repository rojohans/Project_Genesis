
import numpy as np

class IcoSphere():
    '''
    The icosphere is made by dividing an icosahedron.
    The initial values of the vertices and faces correspond to a regular icosahedron.
    '''

    def __init__(self,
                 numberOfDivisions = 0):

        # A dictionary used to prevent duplicate vertices.
        self.middle_point_cache = {}

        # Golden ratio
        PHI = (1 + np.sqrt(5)) / 2
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





