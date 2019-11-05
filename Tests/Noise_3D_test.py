import numpy as np
import Templates.Templates

import Simulation.Noise

import Visualization

from scipy import interpolate
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Delaunay

from tvtk.api import tvtk
from mayavi import mlab

import pickle
import scipy.spatial



#                                                                        gridSize, variability, roughness/smoothness
#finePoints, basicNoise, interpolatedValues, generatedNoise = Simulation.Noise.FractalNoise3D(gridSize, 2, 2)


#gridSize = 256
numberOfThetaValues = 300  # Used as the 2D resolution (# theta values, the # phi values is twice this number.)
faceColorSmoothing = False   # False: Each face will have a single colour based of the corner values.
                             #  True: The face colour varies throughout the face, interpolated from the corner values.
projectTopography = True     # False: The heightvalues of the points will be used to illustrate height.
                             #  True: The surface is projected on a spherical shell.
fractalNoise = False # Determines wether a combination of noises should be used, or juts one.
CalculateFlow = True # Determines if the vertex noise should be converted into a vector field.

# For division values of 8 or greater the visualization becomes slow and laggy. (On mac)
import time
tic = time.clock()
#world = Simulation.Templates.IcoSphere(6)

world = Templates.Templates.GetIcoSphere(6)
# IcoSphereSimple creates an icosphere without the neighbour lists.
#world = Templates.Templates.IcoSphereSimple(6)



# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
# Consider saving the IcoSphere data to a .csv file instead of a .pkl file. This could improve load times.
# The Neighbour object within the ico sphere takes up considerable amounts of memory. Consider just calculating the
# neighbours for one distance value, instead of a range.
# Consider saving the ico sphere as multiple .csv files.
# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
toc = time.clock()
print(toc-tic)



vertsArray = world.vertices
facesArray = world.faces

print('#vertices = ', np.size(vertsArray, 0))
print('#faces', np.size(facesArray, 0))
print('Subdivision done')

b = 1.8 # Used as fractal multiplier. A Higher fractal multiplier gives a smoother noise space.

if fractalNoise is True:
    interpolatedRadius = Simulation.Noise.PerlinNoiseSpherical(256,
                                                               vertsArray.copy(),
                                                               3,
                                                               1.5)/b**7
    interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(128,
                                                               vertsArray.copy(),
                                                               3,
                                                               1.5)/b**6
    interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(64,
                                                                vertsArray.copy(),
                                                                3,
                                                                1.5)/b**5
    interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(32,
                                                                vertsArray.copy(),
                                                                3,
                                                                1.5)/b**4
    interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(16,
                                                                vertsArray.copy(),
                                                                3,
                                                                1.5)/b**3
    interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(8,
                                                                vertsArray.copy(),
                                                                3,
                                                                1.5)/b**2
    #interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(4,
    #                                                            vertsArray.copy(),
    #                                                            3,
    #                                                            1.5)/b
    #interpolatedRadius += Simulation.Noise.PerlinNoiseSpherical(2,
    #                                                            vertsArray.copy(),
    #                                                            3,
    #                                                            1.5)
else:
    #interpolatedRadius = Simulation.Noise.PerlinNoiseSpherical(16,
    #                                                            vertsArray.copy(),
    #                                                            3,
    #                                                            1.5)
    tic = time.clock()
    world.radius = Simulation.Noise.PerlinNoiseSpherical(8,
                                                         world.vertices.copy(),
                                                         numberOfInitialIterationsToSkip = 2,
                                                         amplitudeScaling = 1.5)
    toc = time.clock()
    print('Noise generation time: ', toc - tic)



xFlow, yFlow, zFlow = Simulation.Noise.PerlinNoise3DFlow(4,
                                                         world.vertices.copy(),
                                                         1,
                                                         1.5,
                                                         projectOnSphere = True,
                                                         normalizedVectors = True)
'''
xFlow /= 10
yFlow /= 10
zFlow /= 10


print(np.shape(world.vertices))
print(np.shape(xFlow))
print(np.shape( np.reshape(xFlow, (world.numberOfvertices, 1)) ))
clusteringElements = np.append(world.vertices, np.reshape(xFlow, (world.numberOfvertices, 1)), axis=1)
clusteringElements = np.append(clusteringElements, np.reshape(yFlow, (world.numberOfvertices, 1)), axis=1)
clusteringElements = np.append(clusteringElements, np.reshape(zFlow, (world.numberOfvertices, 1)), axis=1)
vertexClusteringKDTree = scipy.spatial.cKDTree(clusteringElements)


p = np.append(world.vertices[10, :], xFlow[10])
p = np.append(p, yFlow[10])
p = np.append(p, zFlow[10])
print(p)

plateID = np.zeros((world.numberOfvertices, 1))
result = vertexClusteringKDTree.query_ball_point(p, 0.1)
plateIDS = result
plateIDSTotal = result
plateID[result] = 1


while np.size(plateIDS) > 0:
    pID = plateIDS[0]
    del plateIDS[0]
    p = np.append(world.vertices[pID, :], xFlow[pID])
    p = np.append(p, yFlow[pID])
    p = np.append(p, zFlow[pID])
    result = vertexClusteringKDTree.query_ball_point(p, 0.1)
    plateID[result] = 1
    plateIDS += result
    plateIDSTotal += result
    print(plateIDS)

print(plateIDS)
'''

'''
plateList = []
numberOfPlates = 100
plateID = -1*np.ones((world.numberOfvertices, 1))
rLimit = 0.7
stepLimit = 600
for iPlate in range(numberOfPlates):
    initialPoint = np.random.randint(0, world.numberOfvertices)
    while plateID[initialPoint] >= 0:
        initialPoint = np.random.randint(0, world.numberOfvertices)
    currentPlateList = [initialPoint]
    initialFlowVector = [xFlow[initialPoint], yFlow[initialPoint], zFlow[initialPoint]]
    iStep = 0
    for platePoint in currentPlateList:
        iStep += 1
        adjacentPoints = world.neighbours.IDList[platePoint][0]
        for p in adjacentPoints:
            r = np.sqrt((initialFlowVector[0] - xFlow[p]) ** 2 +
                        (initialFlowVector[1]-yFlow[p]) ** 2 +
                        (initialFlowVector[2]-zFlow[p]) ** 2)
            if r < rLimit and plateID[p] < 0:
                currentPlateList.append(p)
                plateID[p] = iPlate
        if iStep == stepLimit:
            break
    plateList.append(currentPlateList)
'''




numberOfPlatesTotal = 300
numberOfPlatesEachIteration = 300
plateID = -1*np.ones((world.numberOfvertices, 1))
rLimit = 0.6
stepLimit = 1000

plateIndexLocal = np.zeros((world.numberOfvertices, 1))
plateList = []#[[] for i in range(numberOfPlatesTotal)]



for iRun in range(int(numberOfPlatesTotal/numberOfPlatesEachIteration)):
    #print(np.linspace(iRun * numberOfPlatesEachIteration, (iRun + 1) * numberOfPlatesEachIteration-1,
    #            numberOfPlatesEachIteration))
    plateListLocal = [[] for i in range(numberOfPlatesEachIteration)]
    #for iPlate in np.linspace(iRun * numberOfPlatesEachIteration, (iRun + 1) * numberOfPlatesEachIteration-1,
    #            numberOfPlatesEachIteration):
    for iPlate in range(numberOfPlatesEachIteration):
        initialPoint = [np.random.randint(0, world.numberOfvertices)]
        while plateID[initialPoint] >= 0:
            initialPoint = np.random.randint(0, world.numberOfvertices)
        plateListLocal[int(iPlate)] = initialPoint
        print(iPlate)
        #print(int(iPlate))
    print(plateListLocal)
    for iStep in range(stepLimit):
        #for iPlate, plate in enumerate(plateList):
        for iPlate, iPlateLocal, plate in zip(np.linspace(iRun * numberOfPlatesEachIteration, (iRun + 1) * numberOfPlatesEachIteration-1,
                numberOfPlatesEachIteration), range(numberOfPlatesEachIteration), plateListLocal):
                if int(plateIndexLocal[int(iPlate)]) < np.size(plate):
                    #print(iPlate)
                    #print(int(iPlate))
                    #print(plateIndexLocal[int(iPlate)])
                    #print(int(plateIndexLocal[int(iPlate)]))
                    #print(np.size(plate))
                    #print(plate)
                    #print(plate[int(plateIndexLocal[int(iPlate)])])

                    #print('------------------------')

                    adjacentPoints = world.neighbours.IDList[plate[int(plateIndexLocal[int(iPlate)])]][0]
                    for p in adjacentPoints:
                        r = np.sqrt((xFlow[plate[0]] - xFlow[p]) ** 2 +
                                    (yFlow[plate[0]] - yFlow[p]) ** 2 +
                                    (zFlow[plate[0]] - zFlow[p]) ** 2)
                        if r < rLimit and plateID[p] < 0:
                            plateListLocal[int(iPlateLocal)].append(p)
                            plateID[p] = iPlate
                            #print(iPlate)
                    plateIndexLocal[int(iPlate)] += 1
    plateList.append(plateListLocal)

''''
print(plateList)
print(plateList[0])
print(plateList[0][0])
quit()
for iPlate in range(numberOfPlates):
    initialPoint = np.random.randint(0, world.numberOfvertices)
    while plateID[initialPoint] >= 0:
        initialPoint = np.random.randint(0, world.numberOfvertices)
    currentPlateList = [initialPoint]
    initialFlowVector = [xFlow[initialPoint], yFlow[initialPoint], zFlow[initialPoint]]
    iStep = 0
    for platePoint in currentPlateList:
        iStep += 1
        adjacentPoints = world.neighbours.IDList[platePoint][0]
        for p in adjacentPoints:
            r = np.sqrt((initialFlowVector[0] - xFlow[p]) ** 2 +
                        (initialFlowVector[1]-yFlow[p]) ** 2 +
                        (initialFlowVector[2]-zFlow[p]) ** 2)
            if r < rLimit and plateID[p] < 0:
                currentPlateList.append(p)
                plateID[p] = iPlate
        if iStep == stepLimit:
            break
    plateList.append(currentPlateList)
'''

#print(np.shape(world.neighbours.IDList))

#print(world.neighbours.IDList[initialPoint][0])
#plateID[world.neighbours.IDList[initialPoint][0]] = 1
#plateID[currentPlateList] = 1
print(np.min(plateID))
print(np.max(plateID))

# Visualizes the globe, as projected or not.
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = plateID,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = 'True')

Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = plateID,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = 'True')
'''
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = world.radius.copy(),
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = False,
                             colormap = 'gist_earth')
'''
Visualization.VisualizeFlow(world.vertices,
                            xFlow,
                            yFlow,
                            zFlow,
                            world.faces,
                            newFigure = False,
                            sizeFactor = 0.03)






print('============================')
print('||>- Visualization done -<||')
print('============================')
mlab.show()
quit()

fig0 = mlab.figure()
fig0.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

'''
quiverObject = mlab.quiver3d(vertsArray[:, 0],
                             vertsArray[:, 1],
                             vertsArray[:, 2],
                             vertexXFlow * (0.015 * (0.5 + 1 * np.abs(vertexXFlow))),
                             vertexYFlow * (0.015 * (0.5 + 1 * np.abs(vertexYFlow))),
                             vertexZFlow * (0.015 * (0.5 + 1 * np.abs(vertexZFlow))),
                             resolution=3,
                             scale_factor=1,
                             color=(1, 0, 0),
                             mode='cone')  # 'arrow'
'''
quiverObject = mlab.quiver3d(vertsArray[:, 0],
                             vertsArray[:, 1],
                             vertsArray[:, 2],
                             vertexXFlow,
                             vertexYFlow,
                             vertexZFlow,
                             resolution=3,
                             scale_factor=0.01,
                             color=(1, 0, 0),
                             mode='cone')  # 'arrow'
#quiverObject.glyph.color_mode = 'color_by_scalar'  # Makes the cones coloured based on their scalar value.

print('Visualization done')

mlab.show()
quit()

heightSurface = mlab.triangular_mesh(
    world.vertices[:, 0] * 0.99,
    world.vertices[:, 1] * 0.99,
    world.vertices[:, 2] * 0.99,
    world.faces,
    scalars=interpolatedRadius[:, 0],
    colormap='gist_earth')#terrain
#'gist_heat'
#'gist_earth'
mlab.show()
quit()


# This si the older noise function. It generates alot of artifacts for low frequencies and high smoothness.
#interpolatedRadius = Simulation.Noise.FractalNoiseSpherical(
#    gridSize,
#    vertsArray,
#    4,
#    1.3)
print('noise generated')
vertexPhiAngle = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
vertexThetaAngle = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))



# Transforms the noise into a uniform noise. Different cdf values can be used in order to get nosie with another
# probability distribution, for example chi^2 or gausian.
N = np.size(interpolatedRadius, 0)
import scipy.stats as stats
cdfValues = stats.uniform.cdf(np.linspace(0, 1, N))
#cdfValues = stats.norm.cdf(np.linspace(0, 1, N), 0.5, 0.25)
#cdfValues = stats.gamma.cdf(np.linspace(0, 1, N), 4)
#cdfValues = np.linspace(0, 1, N) # Used for a uniform cdf.
#print(stats.norm.pdf(0, 0.5, 0.1))
#print(stats.norm.pdf(1, 0.5, 0.1))

#print(np.min(cdfValues))
#print(np.max(cdfValues))
cdfValues -= np.min(cdfValues)
cdfValues /= np.max(cdfValues)
interpolatedRadius -= 1.1
#print('-------------------')
for cdf in cdfValues:
    interpolatedRadius[interpolatedRadius == np.min(interpolatedRadius)] = cdf
    #print(cdf)
#print('-------------------')


# Consider using basis function described here: http://weber.itn.liu.se/~stegu/TNM084-2017/worley-originalpaper.pdf

'''
# Used to visualize the cdf of the function.
noiseValues = np.sort(interpolatedRadius, axis = 0)
#print(noiseValues)
constantY = np.zeros((N, 1))
x = np.linspace(0, 1, N)

fig0 = mlab.figure()
fig0.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
mlab.plot3d(np.linspace(0, 1, N), constantY[:, 0], noiseValues[:, 0], tube_radius = None, line_width = 0.5)
#(np.sin(x * np.pi - np.pi / 2) + 1) / 2
print('cdf visualization done')
mlab.show()
quit()
'''


if CalculateFlow:
    #------------------------------------------------------------------------
    # Generates the vector field, the noise value assigned to each vertex is used as the angle of the flow vector of
    # that vertex.
    # noise = 0    --> 0  degrees : Flow in positive phi direction.
    # noise = 0.25 --> 90 degrees : Flow in positive theta direction
    vertexFlowAngle = interpolatedRadius*2*np.pi# + np.pi
    vertexXFlow = - np.cos(vertexFlowAngle[:, 0]) * np.sin(vertexPhiAngle)\
                  + np.sin(vertexFlowAngle[:, 0]) * np.cos(vertexPhiAngle)*np.sin(vertexThetaAngle)
    vertexYFlow = np.cos(vertexFlowAngle[:, 0]) * np.cos(vertexPhiAngle)\
                  + np.sin(vertexFlowAngle[:, 0]) * np.sin(vertexPhiAngle)*np.sin(vertexThetaAngle)
    vertexZFlow = np.sin(vertexFlowAngle[:, 0]) * np.cos(vertexThetaAngle)
    print('Vector field generated')
    #----------------------------------------------------------------



    # ====================================================================================================
    #                            VISUALIZATION OF THE NEIGHBOURLIST
    # ====================================================================================================

    clusteringVertices = np.append(vertsArray, vertexFlowAngle/50, axis = 1)
    vertexClusteringKDTree = scipy.spatial.cKDTree(clusteringVertices)


    clusterID = np.zeros((np.size(vertexFlowAngle), 1))

    for initialPoint in range(45):
        #r = np.random.randint(0, world.numberOfvertices)
        #print(type(world.neighbours.IDList))
        #print(type(world.neighbours.IDList[0]))
        #print(world.neighbours.IDList[0])
        clusterID[list(world.neighbours.IDList[initialPoint])] += 1
    #clusterID[vertexNeighbourIDList[10]] = vertexNeighbourDistanceList[10]

    fig0 = mlab.figure()
    fig0.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

    heightSurface = mlab.triangular_mesh(
        vertsArray[:, 0]*0.99,
        vertsArray[:, 1]*0.99,
        vertsArray[:, 2]*0.99,
        facesArray,
        scalars=interpolatedRadius[:, 0]*0,
        colormap='gist_earth')

    print(np.shape(vertsArray[:, 0]))
    print(np.shape(clusterID[:, 0]))

    pointObject = mlab.points3d(vertsArray[:, 0],
                                vertsArray[:, 1],
                                vertsArray[:, 2],
                                1 + 0.5*clusterID[:, 0],
                                scale_factor = 0.025,
                                scale_mode = 'vector')
    #pointObject.glyph.color_mode = 'color_by_scalar' # Makes the cones coloured based on their scalar value.
    mlab.show()
    quit()
    # ====================================================================================================



    '''
    for iStep in range(10):
        r = np.random.randint(np.size(vertexFlowAngle))
        #print(r)


        #print(clusteringVertices[r, :])
        result = vertexClusteringKDTree.query_ball_point(clusteringVertices[r, :], 0.1)
        #result = vertexClusteringKDTree.query_ball_point(vertexFlowAngle[r, :], 0.7)
        #print(result)

        clusterID[result] = iStep+1
        #print(clusterID)
        #print(np.sum(clusterID))
    #quit()
    '''

    fig0 = mlab.figure()
    fig0.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

    quiverObject = mlab.quiver3d(vertsArray[:, 0],
                                 vertsArray[:, 1],
                                 vertsArray[:, 2],
                                 vertexXFlow * (0.015 * (0.5 + 1*np.abs(vertexXFlow))),
                                 vertexYFlow * (0.015 * (0.5 + 1*np.abs(vertexYFlow))),
                                 vertexZFlow * (0.015 * (0.5 + 1*np.abs(vertexZFlow))),
                                 resolution = 3,
                                 scale_factor = 1,
                                 scalars = clusterID,
                                 color = (1, 0 , 0),
                                 mode = 'cone') # 'arrow'
    quiverObject.glyph.color_mode = 'color_by_scalar' # Makes the cones coloured based on their scalar value.

    heightSurface = mlab.triangular_mesh(
        vertsArray[:, 0]*0.99,
        vertsArray[:, 1]*0.99,
        vertsArray[:, 2]*0.99,
        facesArray,
        scalars=interpolatedRadius[:, 0]*0,
        colormap='gist_earth')

    mlab.show()


    quit()







    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Clustering based on discrete boundaries.
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------


    # 500 gives a better visualization of the thin branches in the trajectory density.
    # numberOfParticles 500/1000
    # Inclusion of frequency 8 or not.

    #
    # Try a flood-fill algorithm for getting the plates, this requires strict borders. A maximum plate radius can be
    # used to avoid the neccesasity of a strict border around the entire plate.
    #

    #
    # Interpolate from several vertices when calculating the acceleration of the flow particles.
    #


    numberOfFlowParticles = 100
    numberOfSteps = 10
    #deltaAngle = np.pi/10
    #deltaVelocity = np.pi/200 # UNUSED, REMOVED 2019-08-19
    maxVelocity = 0.02

    u = np.transpose(np.random.rand(numberOfFlowParticles, 1))
    v = np.transpose(np.random.rand(numberOfFlowParticles, 1))
    flowParticlePhiAngle = 2 * np.pi * u
    flowParticleThetaAngle = np.arccos((2 * v - 1)) - np.pi / 2

    # Stores the trajectory of each particle. [#steps, #particles]
    flowParticlePhiTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
    flowParticlePhiVelocityTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
    flowParticleThetaTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
    flowParticleThetaVelocityTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))

    # Phi and theta values of the flow vectors.
    phiArray = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
    thetaArray = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))

    trajectoryDensity = np.zeros((np.size(interpolatedRadius, 0), 1)) # Used for visualizing the trejectory density.

    import time
    tic = time.clock()
    vertexKDTree = scipy.spatial.cKDTree(vertsArray)

    for phi, theta, iParticle in zip(np.transpose(flowParticlePhiAngle[0]), np.transpose(flowParticleThetaAngle[0]), range(numberOfFlowParticles)):
        phiVelocity = 0
        thetaVelocity = 0

        flowParticlePhiTrajectory[0, iParticle] = phi
        flowParticlePhiVelocityTrajectory[0, iParticle] = 0
        flowParticleThetaTrajectory[0, iParticle] = theta
        flowParticleThetaVelocityTrajectory[0, iParticle] = 0
        for iStep in range(1, numberOfSteps):
            if False:
                arcDistance = np.arccos(np.cos(phi)* np.cos(theta) * vertsArray[:, 0]
                                        + np.sin(phi) * np.cos(theta) * vertsArray[:, 1]
                                        + np.sin(theta) * vertsArray[:, 2])

                trajectoryDensity[arcDistance == np.min(arcDistance)] += 1
                localFlowAngle = vertexFlowAngle[arcDistance == np.min(arcDistance)][0]

                phiVelocity = (2*phiVelocity + np.cos(localFlowAngle[0])*maxVelocity)/3
                #phiVelocity = np.cos(localFlowAngle)/50
                thetaVelocity = (2*thetaVelocity + np.sin(localFlowAngle[0])*maxVelocity)/3
                #thetaVelocity = np.sin(localFlowAngle)/50
            else:
                flowCoordinates = [np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)]
                queryresult = vertexKDTree.query(flowCoordinates)

                trajectoryDensity[queryresult[1]] += 1
                localFlowAngle = vertexFlowAngle[queryresult[1]][0]

                #phiVelocity = (phiVelocity + 2*np.cos(localFlowAngle)*maxVelocity)/3
                phiVelocity = np.cos(localFlowAngle)/50
                #thetaVelocity = (thetaVelocity + 2*np.sin(localFlowAngle)*maxVelocity)/3
                thetaVelocity = np.sin(localFlowAngle)/50

            # -pi/2 : pi/2
            phi -= phiVelocity
            theta += thetaVelocity
            if theta < -np.pi/2 or theta > np.pi/2:
                phi += np.pi
                theta = np.sign(theta) * np.pi - theta

                phiVelocity = -phiVelocity # ??????
                thetaVelocity = -thetaVelocity
                #print('***')

            # Saves velocities and coordinates in arrays.
            flowParticlePhiVelocityTrajectory[iStep, iParticle] = phiVelocity
            flowParticleThetaVelocityTrajectory[iStep, iParticle] = thetaVelocity
            flowParticlePhiTrajectory[iStep, iParticle] = phi
            flowParticleThetaTrajectory[iStep, iParticle] = theta
        #print('----------------------------------------------------------------')



    xTrajectory = np.cos(flowParticlePhiTrajectory) * np.cos(flowParticleThetaTrajectory)
    yTrajectory = np.sin(flowParticlePhiTrajectory) * np.cos(flowParticleThetaTrajectory)
    zTrajectory = np.sin(flowParticleThetaTrajectory)

    trajectoryDensity[trajectoryDensity <= 2] = 0
    #trajectoryDensity[trajectoryDensity>10] = 10
    trajectoryDensity = np.sqrt(trajectoryDensity)

    toc = time.clock()
    print(toc-tic)
    print('Trajectories calculated')

    fig0 = mlab.figure()
    fig0.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()




    #dx = - flowParticlePhiVelocityTrajectory[:, 0] * np.sin(flowParticlePhiTrajectory[:, 0]) \
    #     + flowParticleThetaVelocityTrajectory[:, 0] * np.cos(flowParticlePhiTrajectory[:, 0]) * np.sin(
    #    flowParticleThetaTrajectory[:, 0])
    #dy = flowParticlePhiVelocityTrajectory[:, 0] * np.cos(flowParticlePhiTrajectory[:, 0]) \
    #     + flowParticleThetaVelocityTrajectory[:, 0] * np.sin(flowParticlePhiTrajectory[:, 0]) * np.sin(
    #    flowParticleThetaTrajectory[:, 0])
    #dz = flowParticleThetaVelocityTrajectory[:, 0] * np.cos(flowParticleThetaTrajectory[:, 0])


    #mlab.quiver3d(xTrajectory[:, 0],
    #              yTrajectory[:, 0],
    #              zTrajectory[:, 0],
    #              dx,
    #              dy,
    #              dz)


    #mlab.plot3d(xTrajectory[:, 0],
    #            yTrajectory[:, 0],
    #            zTrajectory[:, 0],
    #            tube_radius = None,
    #            line_width = 0.5)
    for iParticle in range(numberOfFlowParticles):
        mlab.plot3d(xTrajectory[:, iParticle],
                    yTrajectory[:, iParticle],
                    zTrajectory[:, iParticle],
                    range(numberOfSteps),
                    tube_radius = None,
                    line_width = 0.5,
                    colormap = 'gray')
        #

    #vertexXFlow * (0.015 * (0.5 + 1 * np.abs(vertexZFlow))),
    #vertexYFlow * (0.015 * (0.5 + 1 * np.abs(vertexZFlow))),
    #vertexZFlow * (0.015 * (0.5 + 1 * np.abs(vertexZFlow))),

    quiverObject = mlab.quiver3d(vertsArray[:, 0],
                                 vertsArray[:, 1],
                                 vertsArray[:, 2],
                                 vertexXFlow * (0.015 * (0.5 + 1*np.abs(vertexXFlow))),
                                 vertexYFlow * (0.015 * (0.5 + 1*np.abs(vertexYFlow))),
                                 vertexZFlow * (0.015 * (0.5 + 1*np.abs(vertexZFlow))),
                                 resolution = 3,
                                 scale_factor = 1,
                                 scalars = vertsArray[:, 2],
                                 color = (1, 0 , 0),
                                 mode = 'cone') # 'arrow'
    #quiverObject.glyph.color_mode = 'color_by_scalar' # Makes the cones coloured based on their scalar value.

    heightSurface = mlab.triangular_mesh(
        vertsArray[:, 0]*0.99,
        vertsArray[:, 1]*0.99,
        vertsArray[:, 2]*0.99,
        facesArray,
        scalars=interpolatedRadius[:, 0]*0,
        colormap='gist_earth')

    #==========================================
    # Visualizes the trajectory density over the sphere in the 2D plane.


    #print(vertexFlowAngle)
    #print(np.shape(vertexFlowAngle))
    #print(np.min(vertexFlowAngle))
    #print(np.max(vertexFlowAngle))

    #vertexKDTree.query()


    phiMesh, thetaMesh = np.meshgrid(
        np.linspace(0, 2*np.pi, 2*numberOfThetaValues),
        np.linspace(-np.pi/2, np.pi/2, numberOfThetaValues))
    phiMesh = np.transpose(phiMesh)
    thetaMesh = np.transpose(thetaMesh)

    phiArray = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
    thetaArray = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))
    initialPoint = np.sqrt(vertsArray[:, 0] ** 2 + vertsArray[:, 1] ** 2 + vertsArray[:, 2] ** 2)

    #interpolatedRadius
    #radiusMesh = interpolate.griddata((phiArray, thetaArray), particleID, (phiMesh, thetaMesh))
    radiusMesh = interpolate.griddata((phiArray, thetaArray), trajectoryDensity, (phiMesh, thetaMesh))
    radiusMesh = radiusMesh[:, :, 0]
    #vertsArray[:, 0:2]

    fig3 = mlab.figure()
    fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


    mlab.mesh(phiMesh, thetaMesh, 0*radiusMesh, scalars = radiusMesh, colormap = 'gist_earth')
    # ==========================================

    print('Visualization done')
    mlab.show()

    quit()
# ------------------------------------------------------------------------


'''
#======================================================================================
#================= UNSURE IF THIS IS NEEDED. WAS REMOVED 2019-08-19 ===================
#======================================================================================
numberOfHotSpots = 20
hotSpotID = np.linspace(0, numberOfHotSpots - 1, numberOfHotSpots)

u = np.transpose(np.random.rand(numberOfHotSpots, 1))
v = np.transpose(np.random.rand(numberOfHotSpots, 1))
hotspotPhiAngle = 2 * np.pi * u
hotspotThetaAngle = np.arccos((2 * v - 1)) - np.pi / 2

# [x, y, z]
hotSpotCoordinates = np.zeros((numberOfHotSpots, 3))
hotSpotCoordinates[:, 0] = np.cos(hotspotPhiAngle) * np.cos(hotspotThetaAngle)
hotSpotCoordinates[:, 1] = np.sin(hotspotPhiAngle) * np.cos(hotspotThetaAngle)
hotSpotCoordinates[:, 2] = np.sin(hotspotThetaAngle)


particleID = np.zeros([np.size(vertsArray, 0), 1])
crustThickness = np.zeros([np.size(vertsArray, 0), 1])
for point, iPoint in zip(vertsArray, range(0, np.size(vertsArray, 0))):
    # The arcangles are computed, these correspond directly to the arcdistances if the radius is 1.
    arcDistance = np.arccos(point[0] * hotSpotCoordinates[:, 0]
                            + point[1] * hotSpotCoordinates[:, 1]
                            + point[2] * hotSpotCoordinates[:, 2])
    particleID[iPoint] = hotSpotID[arcDistance == np.min(arcDistance)]
    a = np.exp(-15*arcDistance**2)
    crustThickness[iPoint] = - np.sum(a)

crustThickness -= np.min(crustThickness)
crustThickness /= np.max(crustThickness)
crustThickness += 1
'''

# The radiuses are adjusted.
radiusModifier = 15.8
#vertexRadius = radiusModifier + interpolatedRadius + 0*crustThickness
vertexRadius = radiusModifier + interpolatedRadius
#print(crustThickness)
print(interpolatedRadius)
print(np.min(interpolatedRadius))
print(np.max(interpolatedRadius))

vertsArray[:, 0] *= vertexRadius[:, 0]
vertsArray[:, 1] *= vertexRadius[:, 0]
vertsArray[:, 2] *= vertexRadius[:, 0]

fig1 = mlab.figure()
fig1.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(hotSpotCoordinates[:, 0]*1.5, hotSpotCoordinates[:, 1]*1.5, hotSpotCoordinates[:, 2]*1.5,
#              color=(1, 1, 1), scale_factor = 0.1)

# ========================================================================
# ------------------------------------------------------------------------
# ========================================================================
# ------------------------------------------------------------------------
# ========================================================================
if faceColorSmoothing is False:
    # When true the colour for each triangle will be varied throughout the triangle. Interpolation is done based on
    # the vertex values. When the statement is false each triangle will have a single colour. The colour is based on
    # the mean of the vertex values.

    #faceHeight = np.mean(interpolatedRadius[facesArray, 0], 1)
    #faceHeight = np.max(particleID[facesArray, 0], 1)
    faceHeight = np.mean(vertexRadius[facesArray, 0], 1)

    if projectTopography is True:
        mesh = mlab.triangular_mesh(vertsArray[:, 0]/vertexRadius[:, 0],
                                vertsArray[:, 1]/vertexRadius[:, 0],
                                vertsArray[:, 2]/vertexRadius[:, 0],
                                facesArray,
                                representation='wireframe',
                                opacity=0,
                                colormap = 'terrain')
    else:
        mesh = mlab.triangular_mesh(vertsArray[:, 0],
                                    vertsArray[:, 1],
                                    vertsArray[:, 2],
                                    facesArray,
                                    representation='wireframe',
                                    opacity=0,
                                    colormap = 'terrain')

    mesh.mlab_source.dataset.cell_data.scalars = faceHeight
    mesh.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
    mesh.mlab_source.update()

    mesh2 = mlab.pipeline.set_active_attribute(mesh,
                                               cell_scalars='Cell data')
    mlab.pipeline.surface(mesh2, colormap = 'gist_earth')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
else:
    heightSurface = mlab.triangular_mesh(
        vertsArray[:, 0]/vertexRadius[:, 0],
        vertsArray[:, 1]/vertexRadius[:, 0],
        vertsArray[:, 2]/vertexRadius[:, 0],
        facesArray,
        scalars = vertexRadius[:, 0],
        colormap='gist_earth')

phiMesh, thetaMesh = np.meshgrid(
    np.linspace(0, 2*np.pi, 2*numberOfThetaValues),
    np.linspace(-np.pi/2, np.pi/2, numberOfThetaValues))
phiMesh = np.transpose(phiMesh)
thetaMesh = np.transpose(thetaMesh)

phiArray = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
thetaArray = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))
initialPoint = np.sqrt(vertsArray[:, 0] ** 2 + vertsArray[:, 1] ** 2 + vertsArray[:, 2] ** 2)

#interpolatedRadius
#radiusMesh = interpolate.griddata((phiArray, thetaArray), particleID, (phiMesh, thetaMesh))
radiusMesh = interpolate.griddata((phiArray, thetaArray), vertexRadius, (phiMesh, thetaMesh))
radiusMesh = radiusMesh[:, :, 0]
#vertsArray[:, 0:2]

fig3 = mlab.figure()
fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

'''
#======================================================================================
#=============== SAVES THE PHI, THETA AND RADIUS DATA TO A .TXT FILE ==================
#======================================================================================
phiArray = phiArray.reshape((np.size(phiArray), 1))
thetaArray = thetaArray.reshape((np.size(thetaArray), 1))
phiArray *= 180/np.pi
phiArray -= 180
thetaArray *= 180/np.pi
a = np.concatenate((phiArray, thetaArray, vertexRadius), axis=1)
np.savetxt('test.txt',a, delimiter=',')
#======================================================================================
'''

mlab.mesh(phiMesh, thetaMesh, 0*radiusMesh, scalars = radiusMesh, colormap = 'gist_earth')

print('visualization done')

mlab.show()

quit()












# An attempt is made to read a template from file, if successful point coordinates and surface simplices will be
# used. If this can not be done the points and simplices are calculated from scratch.
try:
    worldTemplate = pickle.load(open('/Users/robinjohansson/Desktop/Project_Genesis/' + 'Templates/' + 'World_Template_' + str(numberOfThetaValues) + '.pkl', 'rb'))

    particleCoordinates = np.zeros((worldTemplate.numberOfPoints, 6))
    particleCoordinates[:, 0:3] = worldTemplate.cartesianCoordinates
    particleCoordinates[:, 3] = worldTemplate.sphericalCoordinates[:, 2]
    particleCoordinates[:, 4:6] = worldTemplate.sphericalCoordinates[:, 0:2]
    surfaceSimplices = worldTemplate.surfaceSimplices
except:
    print('--------------- WARNING ----------------\n'
          '------- A TEMPLATE WAS NOT FOUND -------\n'
          '--------------- WARNING ----------------\n'
          'The following calculations may take long time depending on the number of points')

    thetaVector = np.linspace(-np.pi / 2, np.pi / 2, numberOfThetaValues)
    dTheta = thetaVector[1] - thetaVector[0]
    numberOfParticles = int(
        np.sum(np.fmax(np.round(np.cos(thetaVector) * 2 * np.pi / dTheta), np.ones(np.shape(thetaVector)))))
    # particleCoordinates = [x, y, z, r, phi, theta]
    particleCoordinates = np.zeros((numberOfParticles, 6))
    iParticle = 0
    for theta in thetaVector:
        phiVector = np.linspace(0, 2 * np.pi * (1 - 1 / np.max((np.round(np.cos(theta) * 2 * np.pi / dTheta), 1))),
                                np.max((np.round(np.cos(theta) * 2 * np.pi / dTheta), 1)))
        for phi in phiVector:
            initialPoint = 1
            particleCoordinates[iParticle, 0] = initialPoint * np.cos(phi) * np.cos(theta)
            particleCoordinates[iParticle, 1] = initialPoint * np.sin(phi) * np.cos(theta)
            particleCoordinates[iParticle, 2] = initialPoint * np.sin(theta)
            particleCoordinates[iParticle, 3] = initialPoint
            particleCoordinates[iParticle, 4] = phi
            particleCoordinates[iParticle, 5] = theta
            iParticle += 1
    # THIS PART MAY TAKE LONG TIME.
    # For numberOfThetaValues > 150 the calculations may take a long time: 30+ min
    # incremental = True enables new points to be added, useful when the tectonic activity creates new land masses.
    # The neighbor information can be used when nearby points are to be found, useful when doing erosion/deposition.
    tri = Delaunay(particleCoordinates[:, 0:3])

'============================================================'


interpolatedRadius = Simulation.Noise.FractalNoiseSpherical(
    gridSize,
    particleCoordinates[:, 0:3],
    3,
    1.7)

# The radiuses are adjusted.
radiusModifier = 19.8
particleCoordinates[:, 0] *= radiusModifier + interpolatedRadius[:, 0]
particleCoordinates[:, 1] *= radiusModifier + interpolatedRadius[:, 0]
particleCoordinates[:, 2] *= radiusModifier + interpolatedRadius[:, 0]
particleCoordinates[:, 3] = radiusModifier +  interpolatedRadius[:, 0]



# 3D globe
fig1 = mlab.figure()
fig1.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], scale_factor=0.03,
#              color=(1, 1, 1))
heightSurface = mlab.triangular_mesh(particleCoordinates[:, 0]/particleCoordinates[:, 3],
                                     particleCoordinates[:, 1]/particleCoordinates[:, 3],
                                     particleCoordinates[:, 2]/particleCoordinates[:, 3],
                                     surfaceSimplices,
                                     scalars = particleCoordinates[:, 3],
                                     colormap = 'gist_earth') # colormap = 'terrain' / 'gist_earth'


#numberOfThetaValues = 50
# 2D projection
phiMesh, thetaMesh = np.meshgrid(
    np.linspace(0, 2*np.pi, 2*numberOfThetaValues),
    np.linspace(-np.pi/2, np.pi/2, numberOfThetaValues))
phiMesh = np.transpose(phiMesh)
thetaMesh = np.transpose(thetaMesh)
radiusMesh = interpolate.griddata(particleCoordinates[:, 4:6], particleCoordinates[:, 3], (phiMesh, thetaMesh))


fig3 = mlab.figure()
fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
mlab.mesh(phiMesh, thetaMesh, 0*radiusMesh, scalars = radiusMesh, colormap = 'gist_earth')


#fig4 = mlab.figure()
#fig4.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(particleCoordinates[:, 4], particleCoordinates[:, 5], particleCoordinates[:, 3], particleCoordinates[:, 3],
#              scale_factor=0.005, colormap = 'gist_earth')

#fig2 = mlab.figure()
#fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(finePoints[:, 0], finePoints[:, 1], finePoints[:, 2],
#                generatedNoise[finePoints[:, 0], finePoints[:, 1], finePoints[:, 2]], scale_factor=1.5, opacity = 1)

print('--------------------------------------')
print('DONE')
mlab.show()







