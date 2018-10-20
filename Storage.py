class World():
    '''
    The World class contains everything which is distinctive of a created world, like the different heightmaps but also
    parameters used to create the world.
    '''

    def __init__(self,
                 initialRockMap,
                 initialSedimentMap,
                 initialTotalMap,
                 rockMap,
                 sedimentMap,
                 totalMap,
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

        # Processed heigthmaps
        self.rockMap = rockMap
        self.sedimentMap = sedimentMap
        self.totalHeightMap = totalMap

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