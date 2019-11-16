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

    #world.radius = Simulation.Noise.PerlinNoiseSpherical(8,
    #                                                     world.vertices.copy(),
    #                                                     numberOfInitialIterationsToSkip = 2,
    #                                                     amplitudeScaling = 1.5)
    pass


#tic = time.clock()
#world.radius = Simulation.Noise.PerlinNoiseSpherical(256,
#                                                     world.vertices.copy(),
#                                                     numberOfInitialIterationsToSkip = 2,
#                                                     amplitudeScaling = 1.5)
'''
toc = time.clock()
print('Noise generation time: ', toc - tic)


# Visualizes the globe, as projected or not.
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = world.radius.copy(),
                             projectTopography = False,
                             projectRadiusSpan = [1, 1.1],
                             interpolatedTriangleColor = False,
                             colormap = 'gist_earth',
                             randomColormap = False)
# Visualizes the globe, as projected or not.
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = world.radius.copy(),
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.1],
                             interpolatedTriangleColor = False,
                             colormap = 'gist_earth',
                             randomColormap = False)
print('done')
mlab.show()
quit()
'''

'''
xFlow, yFlow, zFlow = Simulation.Noise.PerlinNoise3DFlow(64,
                                                         world.vertices.copy(),
                                                         1,
                                                         amplitudeScaling = 3,
                                                         projectOnSphere = True,
                                                         normalizedVectors = True)
Visualization.VisualizeFlow(world.vertices,
                            xFlow,
                            yFlow,
                            zFlow,
                            world.faces,
                            newFigure = True,
                            sizeFactor = 0.03)
xFlow, yFlow, zFlow = Simulation.Noise.PerlinNoise3DFlow(64,
                                                         world.vertices.copy(),
                                                         1,
                                                         amplitudeScaling = 1.2,
                                                         projectOnSphere = True,
                                                         normalizedVectors = True)
Visualization.VisualizeFlow(world.vertices,
                            xFlow,
                            yFlow,
                            zFlow,
                            world.faces,
                            newFigure = True,
                            sizeFactor = 0.03)
print('done')
mlab.show()
quit()
'''

# Kolla upp hur tektoniska plattor  har varierat över tid.


if False:

    mainPoint = world.vertices[np.random.randint(0, world.numberOfvertices), :]
    candidatePoint = world.vertices[np.random.randint(0, world.numberOfvertices), :]

    mainTheta = np.arccos(mainPoint[2])
    candidateTheta = np.arccos(candidatePoint[2])
    mainPhi = np.arctan(mainPoint[0] / mainPoint[1]) + np.pi * (np.abs(mainPoint[0]) - mainPoint[0]) / 2
    candidatePhi = np.arctan(candidatePoint[0] / candidatePoint[1]) + np.pi * (np.abs(candidatePoint[0]) - candidatePoint[0]) / 2

    #thetaVector = np.array([20, 30]).transpose()*np.pi/180
    #phiVector = np.array([30, 50]).transpose()*np.pi/180
    thetaVector = np.array([mainTheta, candidateTheta]).transpose()*np.pi/180
    phiVector = np.array([mainPhi, candidatePhi]).transpose()*np.pi/180


    xV = np.cos(phiVector) * np.cos(thetaVector)
    yV = np.sin(phiVector) * np.cos(thetaVector)
    zV = np.sin(thetaVector)

    xP = np.array([mainPoint[0], candidatePoint[0]]).transpose()
    yP = np.array([mainPoint[1], candidatePoint[1]]).transpose()
    zP = np.array([mainPoint[2], candidatePoint[2]]).transpose()
    xP = np.reshape(xP, (2, 1))
    yP = np.reshape(yP, (2, 1))
    zP = np.reshape(zP, (2, 1))
    #xP = np.reshape(np.cos(phiVector) * np.sin(thetaVector), (2, 1))
    #yP = np.reshape(np.sin(phiVector) * np.sin(thetaVector), (2, 1))
    #zP = np.reshape(np.cos(thetaVector), (2, 1))

    print([xV[0], yV[0], zV[0]])
    print([xV[1], yV[1], zV[1]])
    #print(np.shape(xP))

    p = np.append(xP, yP, axis = 1)
    p = np.append(p, zP, axis = 1)


    dTheta = thetaVector[0] - thetaVector[1]
    dPhi = phiVector[1] - phiVector[0]

    ATheta = np.array([[np.cos(dTheta), -np.sin(dTheta)], [np.sin(dTheta), np.cos(dTheta)]])
    APhi = np.array([[np.cos(dPhi), -np.sin(dPhi)], [ np.sin(dPhi), np.cos(dPhi)]])

    vTheta = np.array([zV[0], np.sqrt(xV[0]**2 + yV[0]**2)]).transpose() # Vector to rotate
    vThetaPrime = ATheta.dot(vTheta) # Rotated vector

    xV[0] = xV[0] * vThetaPrime[1]/vTheta[1]
    yV[0] = yV[0] * vThetaPrime[1]/vTheta[1]
    zV[0] = vThetaPrime[0]

    vPhi = np.array([xV[0], yV[0]]).transpose()
    vPhiPrime = APhi.dot(vPhi)

    xV[0] = vPhiPrime[0]
    yV[0] = vPhiPrime[1]
    p[1, :] = p[0, :]

    print('--------------------')
    print([xV[0], yV[0], zV[0]])
    print([xV[1], yV[1], zV[1]])

    Visualization.VisualizeFlow(p,
                                xV,
                                yV,
                                zV,
                                newFigure = False,
                                sizeFactor = 0.03)
    print('done')
    mlab.show()
    quit()

# Generates the perlin flow, the function returns 3 distinct vectors.
xFlow, yFlow, zFlow = Simulation.Noise.PerlinNoise3DFlow(4,
                                                         world.vertices.copy(),
                                                         1,
                                                         amplitudeScaling = 1.5,
                                                         projectOnSphere = True,
                                                         normalizedVectors = True)


identityMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
numberOfSets = 6
numberOfPlatesEachSet = 200
plateID = -1*np.ones((world.numberOfvertices, 1))
rLimit = 0.4
stepLimit = 500

plateIndexLocal = np.zeros((world.numberOfvertices, 1))
plateList = []#[[] for i in range(numberOfPlatesTotal)]


availablePoints = [i for i in np.linspace(0, world.numberOfvertices-1, world.numberOfvertices, dtype=int)]


for iRun in range(numberOfSets):
    plateListSet = [[] for i in range(numberOfPlatesEachSet)]
    for iPlateSet in range(numberOfPlatesEachSet):
        initialPoint = [np.random.randint(0, world.numberOfvertices)]

        if np.size(availablePoints) > 0:
            initialPoint = np.random.choice(availablePoints)
            availablePoints.remove(initialPoint)
        else:
            # No more plates should be created.
            break

        plateListSet[int(iPlateSet)] = [initialPoint]
        #print(initialPoint)
        plateID[initialPoint] = iRun*numberOfPlatesEachSet + iPlateSet
    #print(plateListSet)
    print('==========================================================')
    #quit()
    for iStep in range(stepLimit):
        for iPlate, iPlateSet, plate in zip(np.linspace(iRun * numberOfPlatesEachSet, (iRun + 1) * numberOfPlatesEachSet-1,
                numberOfPlatesEachSet), range(numberOfPlatesEachSet), plateListSet):
                if int(plateIndexLocal[int(iPlate)]) < np.size(plate):
                    '''
                    print(iPlateSet)
                    print(iPlate)
                    print(int(iPlate))
                    print(plateIndexLocal[int(iPlate)])
                    print(int(plateIndexLocal[int(iPlate)]))
                    print(np.size(plate))
                    print(plate)
                    print(plate[int(plateIndexLocal[int(iPlate)])])

                    print('------------------------')
                    '''
                    adjacentPoints = world.neighbours.IDList[plate[int(plateIndexLocal[int(iPlate)])]][0]
                    for p in adjacentPoints:

                        # x-/y-/zFlow should be rotated before calculating r

                        xFlow = np.reshape(xFlow, (world.numberOfvertices, 1))
                        yFlow = np.reshape(yFlow, (world.numberOfvertices, 1))
                        zFlow = np.reshape(zFlow, (world.numberOfvertices, 1))
                        flowVectors = np.append(xFlow, yFlow, axis=1)
                        flowVectors = np.append(flowVectors, zFlow, axis=1)

                        # The main point, and the candidate point which might be added.
                        mainPoint = world.vertices[plate[0], :].copy()
                        candidatePoint = world.vertices[p, :].copy()

                        mainTheta = np.arcsin(mainPoint[2])
                        candidateTheta = np.arcsin(candidatePoint[2])
                        if mainPoint[0]==0:
                            # Take care of /0 case. The phi angle should be -pi/2 or pi/2 depending on the y-value.
                            if mainPoint[1]>0:
                                mainPhi = np.pi/2
                            else:
                                mainPhi = -np.pi/2
                        else:
                            mainPhi = np.arctan(mainPoint[1] / mainPoint[0]) + np.pi * (
                                1 - np.sign(mainPoint[0])) / 2
                        if candidatePoint[0]==0:
                            # Take care of /0 case. The phi angle should be -pi/2 or pi/2 depending on the y-value.
                            if candidatePoint[1]>0:
                                candidatePhi = np.pi/2
                            else:
                                candidatePhi = -np.pi/2
                        else:
                            candidatePhi = np.arctan(candidatePoint[1] / candidatePoint[0]) + np.pi * (
                                1 - np.sign(candidatePoint[0])) / 2

                        # Flow vectors. The candidate flow vector will be rotated to the position of the main flow vector.
                        mainFlow = flowVectors[plate[0], :].copy()
                        candidateFlow = flowVectors[p, :].copy()

                        dTheta = mainTheta - candidateTheta
                        dPhi = mainPhi - candidatePhi

                        ATheta = np.array([[np.cos(dTheta), -np.sin(dTheta)], [np.sin(dTheta), np.cos(dTheta)]])
                        APhi = np.array([[np.cos(dPhi), -np.sin(dPhi)], [np.sin(dPhi), np.cos(dPhi)]])

                        # Create method for rotating a vector. input: {dTheta, dPhi, vector}

                        rhoCandidate = np.sqrt(candidatePoint[0] ** 2 + candidatePoint[1] ** 2)
                        rhoMain = np.sqrt(mainPoint[0] ** 2 + mainPoint[1] ** 2)
                        if rhoCandidate == 0:
                            midpoint = np.array([0, 0, mainPoint[2]])
                        else:
                            midPoint = np.array(
                                [candidatePoint[0] * rhoMain / rhoCandidate, candidatePoint[1] * rhoMain / rhoCandidate,
                                 mainPoint[2]])

                        # The crossVector specifies the axes around which the rotation will occur.
                        crossVector = np.cross(candidatePoint, midPoint)

                        skewMatrix = np.array([[0, -crossVector[2], crossVector[1]],
                                               [crossVector[2], 0, -crossVector[0]],
                                               [-crossVector[1], crossVector[0], 0]])
                        skewMatrixSquared = np.dot(skewMatrix, skewMatrix)

                        s = np.sqrt(crossVector[0] ** 2 + crossVector[1] ** 2 + crossVector[2] ** 2)
                        c = np.dot(midPoint, candidatePoint)
                        R = identityMatrix + skewMatrix + skewMatrixSquared * (1 - c) / (s ** 2 + 0.000001)

                        vectorToRotate = candidatePoint + candidateFlow
                        vLength = np.sqrt(vectorToRotate[0] ** 2 + vectorToRotate[1] ** 2 + vectorToRotate[2] ** 2)
                        vectorToRotate /= vLength

                        rotatedVector = np.dot(R, vectorToRotate)

                        rotatedVector *= vLength
                        rotatedVector -= midPoint

                        vPhi = np.array([rotatedVector[0], rotatedVector[1]]).transpose()
                        vPhiPrime = APhi.dot(vPhi)

                        candidateFlow[0] = vPhiPrime[0]
                        candidateFlow[1] = vPhiPrime[1]
                        candidateFlow[2] = rotatedVector[2]





                        '''
                        #mainPoint = world.vertices[plate[0], :].copy()
                        #candidatePoint = world.vertices[p, :].copy()
                        # mainPoint = world.vertices[1337, :].copy()
                        # candidatePoint = world.vertices[137, :].copy()
                        #                        mainFlow = flowVectors[plate[0], :].copy()
                        #                        candidateFlow = flowVectors[p, :].copy()
                        a = [i for i in range(world.numberOfvertices)]

                        N = 1000

                        correctPhiList = []
                        correctThetaList = []
                        incorrectPhiList = []
                        incorrectThetaList = []


                        for i in range(N):
                            b = world.vertices[np.random.choice(a, 2), :].copy()
                            #mainPoint = np.array([0.00000001, 0, -1])
                            #candidatePoint = np.array([1, 0, 0])
                            mainPoint = b[0, :]
                            candidatePoint = b[1, :]

                            mainTheta = np.arcsin(mainPoint[2])
                            candidateTheta = np.arcsin(candidatePoint[2])
                            mainPhi = np.arctan(mainPoint[1] / mainPoint[0]) + np.pi * (
                            1 - np.sign(mainPoint[0]))/2
                            candidatePhi = np.arctan(candidatePoint[1] / candidatePoint[0]) + np.pi * (
                            1 - np.sign(candidatePoint[0]))/2

                            # Theta unit vectors.
                            mainFlow = np.array([-np.cos(mainPhi)*np.sin(mainTheta), -np.sin(mainPhi)*np.sin(mainTheta), np.cos(mainTheta)])
                            candidateFlow = np.array([-np.cos(candidatePhi) * np.sin(candidateTheta), -np.sin(candidatePhi) * np.sin(candidateTheta),
                                                 np.cos(candidateTheta)])

                            # Phi unit vectors.
                            #mainFlow = np.array([-np.sin(mainPhi), np.cos(mainPhi), 0*np.sin(mainPhi)])
                            #candidateFlow = np.array([-np.sin(candidatePhi), np.cos(candidatePhi), 0*np.sin(mainPhi)])




                            if True:
                                dTheta = mainTheta - candidateTheta
                                dPhi = mainPhi - candidatePhi

                                ATheta = np.array([[np.cos(dTheta), -np.sin(dTheta)], [np.sin(dTheta), np.cos(dTheta)]])
                                APhi = np.array([[np.cos(dPhi), -np.sin(dPhi)], [np.sin(dPhi), np.cos(dPhi)]])

                                # Create method for rotating a vector. input: {dTheta, dPhi, vector}


                                I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

                                rhoCandidate = np.sqrt(candidatePoint[0] ** 2 + candidatePoint[1] ** 2)
                                rhoMain = np.sqrt(mainPoint[0] ** 2 + mainPoint[1] ** 2)
                                midPoint = np.array([candidatePoint[0] * rhoMain / rhoCandidate,
                                                     candidatePoint[1] * rhoMain / rhoCandidate, mainPoint[2]])

                                crossVector = np.cross(candidatePoint, midPoint)

                                skewMatrix = np.array([[0, -crossVector[2], crossVector[1]],
                                                       [crossVector[2], 0, -crossVector[0]],
                                                       [-crossVector[1], crossVector[0], 0]])
                                skewMatrixSquared = np.dot(skewMatrix, skewMatrix)

                                s = np.sqrt(crossVector[0] ** 2 + crossVector[1] ** 2 + crossVector[2] ** 2)
                                c = np.dot(midPoint, candidatePoint)
                                R = I + skewMatrix + skewMatrixSquared * (1 - c) / s ** 2

                                vectorToRotate = candidatePoint + candidateFlow
                                vLength = np.sqrt(
                                    vectorToRotate[0] ** 2 + vectorToRotate[1] ** 2 + vectorToRotate[2] ** 2)
                                vectorToRotate /= vLength

                                rotatedVector = np.dot(R, vectorToRotate)

                                rotatedVector *= vLength
                                rotatedVector -= midPoint

                                vPhi = np.array([rotatedVector[0], rotatedVector[1]]).transpose()
                                vPhiPrime = APhi.dot(vPhi)

                                candidateFlow[0] = vPhiPrime[0]
                                candidateFlow[1] = vPhiPrime[1]
                                candidateFlow[2] = rotatedVector[2]

                                dFlow = mainFlow - rotatedVector
                                dFlowSize = np.sqrt(dFlow[0]**2 + dFlow[1]**2 + dFlow[2]**2)



                            if np.sqrt(candidateFlow[0]**2 + candidateFlow[1]**2 + candidateFlow[2]**2) < 0.95:
                                print('vector is too short.')

                            vectorDiff = mainFlow - candidateFlow
                            diffSize = np.sqrt(vectorDiff[0]**2 + vectorDiff[1]**2 + vectorDiff[2]**2 )
                            if diffSize > 0.0001:
                                # incorrect
                                incorrectPhiList.append(dPhi)
                                incorrectThetaList.append(dTheta)
                                
                                print('main point      = ', mainPoint)
                                print('candidate point = ', candidatePoint)
                                print('main flow       = ', mainFlow)
                                print('candidate flow  = ', candidateFlow)
                                print('main theta      = ', mainTheta)
                                print('candidate theta = ', candidateTheta)
                                print('main phi        = ', mainPhi)
                                print('candidate phi   = ', candidatePhi)
                                print('dTheta          = ', dTheta)
                                print('dPhi            = ', dPhi)
                                print('vector diff     = ', vectorDiff)
                                print('diff size       = ', diffSize)
                                print('----------------')
                                print('----------------')
                                
                            else:
                                #correct
                                correctPhiList.append(dPhi)
                                correctThetaList.append(dTheta)



                        print(np.shape(correctPhiList))
                        print(np.shape(incorrectPhiList))

                        print(np.mean(correctPhiList))
                        print(np.mean(incorrectPhiList))
                        print(np.mean(correctThetaList))
                        print(np.mean(incorrectThetaList))

                        #print(np.max(correctThetaList))
                        #print(np.min(correctThetaList))

                        #print(np.max(incorrectThetaList))
                        #print(np.min(incorrectThetaList))

                        #Visualization.VisualizeFlow(np.append(mainPoint, candidatePoint, axis = 0),
                        #                            [mainFlow[0], candidateFlow[0]],
                        #                            [mainFlow[1], candidateFlow[1]],
                        #                            [mainFlow[2], candidateFlow[2]],
                        #                            newFigure=False,
                        #                            sizeFactor=0.03)
                        #mlab.show()

                        quit()
                        '''


                        #r = np.sqrt((xFlow[plate[0]] - xFlow[p]) ** 2 +
                        #            (yFlow[plate[0]] - yFlow[p]) ** 2 +
                        #            (zFlow[plate[0]] - zFlow[p]) ** 2)
                        # Calculates the norm between the vectors. Will be used to determine if the candidate should be
                        # part of the existing plate.
                        r = np.sqrt((mainFlow[0] - candidateFlow[0]) ** 2 +
                                    (mainFlow[1] - candidateFlow[1]) ** 2 +
                                    (mainFlow[2] - candidateFlow[2]) ** 2)
                        #print(r)
                        #quit()
                        if r < rLimit and plateID[p] < 0:
                            #print(plateID[p])
                            plateListSet[int(iPlateSet)].append(p)
                            plateID[p] = iPlate
                            #print(plate)
                            #print(p)
                            #print(iPlate)
                            #print(iPlateSet)
                            #print('----------')
                            availablePoints.remove(p)
                            #print(iPlate)
                    plateIndexLocal[int(iPlate)] += 1
    plateList.append(plateListSet)
#print(plateList)
print(np.size(availablePoints))
print(np.min(plateID))
print(np.max(plateID))


# The plates should be compared and combined here.


# Visualizes the globe, as projected or not.
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = plateID,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = True)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = plateID,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = True)
Visualization.VisualizeFlow(world.vertices,
                            xFlow[:, 0],
                            yFlow[:, 0],
                            zFlow[:, 0],
                            world.faces,
                            newFigure = False,
                            sizeFactor = 0.03)

print('============================')
print('||>- Visualization done -<||')
print('============================')
mlab.show()
quit()

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







