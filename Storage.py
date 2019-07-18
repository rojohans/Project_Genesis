class World():
    '''
    The World class contains everything which is distinctive of a created world, like the different heightmaps but also
    parameters used to create the world.
    ====================================================================================================================
    ====================================================================================================================
    ============================== NOTE THAT THIS IS A 2D GRID BASED WORLD =============================================
    ====================================================================================================================
    ====================================================================================================================
    '''

    def __init__(self,
                 initialRockMap,
                 initialSedimentMap,
                 initialTotalMap,
                 initialWaterMap,
                 rockMap,
                 sedimentMap,
                 totalMap,
                 waterMap,
                 numberOfRuns,
                 numberOfDrops,
                 numberOfSteps,
                 inertia,
                 capacityMultiplier,
                 depositionRate,
                 erosionRate,
                 erosionRadius,
                 maximumUnimprovedSteps):

        # Initial heigthmaps
        self.initialRockMap = initialRockMap
        self.initialSedimentMap = initialSedimentMap
        self.initialTotalMap = initialTotalMap
        self.initialWaterMap = initialWaterMap

        # Processed heigthmaps
        self.rockMap = rockMap
        self.sedimentMap = sedimentMap
        self.totalHeightMap = totalMap
        self.waterMap = waterMap

        # parameters
        self.numberOfRuns = numberOfRuns
        self.numberOfDrops = numberOfDrops
        self.numberOfSteps = numberOfSteps
        self.inertia = inertia
        self.capacityMultiplier = capacityMultiplier
        self.depositionRate = depositionRate
        self.erosionRate = erosionRate
        self.erosionRadius = erosionRadius
        self.maximumUnimprovedSteps = maximumUnimprovedSteps


class World3DTemplate():
    #
    #
    # This class is used to create templates from which to create new 3D worlds. This is mostly done for performance
    # since the delaunay triangulation takes alot of time for a large number of points, the triangulation is used to get
    # the simplices which connects the points (needed for visualization).
    #
    #

    def __init__(self,
                 cartesianCoordinates,
                 sphericalCoordinates,
                 surfaceSimplices,
                 numberOfPoints,
                 numberOfThetaValues):

        # cartesianCoordinates = [x, y, z]
        # sphericalCoordinates = [phi, theta, r]
        self.cartesianCoordinates = cartesianCoordinates
        self.sphericalCoordinates = sphericalCoordinates
        self.surfaceSimplices = surfaceSimplices

        self.numberOfPoints = numberOfPoints
        self.numberOfThetaValues = numberOfThetaValues



def GetWorldFromFile():
    '''
    The user uses an easygui window to select a .pkl file. From that file an object is retrieved and returned to the user.
    :return:
        An object of the class <Storage.World>
    '''
    import easygui
    fileName = easygui.fileopenbox(default="*.pkl")
    if fileName == None:
        print('-----------------------------\n||| NO FILE WAS SELECTED  |||\n||| THE PROGRAM WILL QUIT |||\n-----------------------------')
        quit()
    else:
        import pickle  # https://docs.python.org/3/library/pickle.html
        return pickle.load(open(fileName, 'rb'))