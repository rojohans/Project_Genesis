import numpy as np
from scipy import interpolate


'''
# THE MAYAVI MODULES ARE ONLY IMPORTED FOR DEBUGGING
'''
import mayavi.mlab
import numpy as np
from mayavi.mlab import *
from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor


def SimpleNoise(gridSize, numberOfInitialIterationsToSkip = 1, amplitudeReduction = 2):
    '''
    # --------ALGORITHM SUMMARY---------
    # An underlying noise array is created with values uniformly distributed between 0 and 1. The underlying noise is
    # sampled at different frequencies and interpolated to give rise to different layers of noise. Each layer is an
    # interpretation of the underlying noise at a certain resolution. The layers are weighted according to their
    # resolution and then combined to form a final multi-layered noise array. The multi-layered noise contain
    # information from all resolutions in varying amount depending on the weights. The noise is naturally periodic.
    #
    # NOTE THAT THIS FUNCTION DO NOT GENERATE PERFECTLY PERIODIC NOISE VALUES.
    #
    # ------------PARAMETERS------------
    # gridSize: [int]
    #   The gridSize indicates the size of the noise array to generate. The value has to be on the form 2^k, where k
    #   is a positive integer.
    #
    # numberOfInitialIterationsToSkip: [int] optional
    #   The initial layers has a great impact on the final noise and may give rise to artifacts, to avoid these
    #   artifacts a set number of initial layers can be left out.
    #   By default the first layer is always skipped since it's constant.
    #
    # amplitudeReduction: [float] optional
    #   By default the amplitude is halved between each layer. Lowering this value will make the final noise more
    #   random. Increasing the value has the opposite effect
    #
    # -------------OUTPUT---------------
    # generatedNoise: [float array]
    #   A [gridSize,gridSize] numpy array containing noise values between 0 and 1. The noise is periodic.
    '''

    samplingInterval = gridSize / pow(2, numberOfInitialIterationsToSkip)
    samplingAmplitude = 1
    generatedNoise = np.zeros([gridSize+1, gridSize+1]) # This array will contain the final multi-layered noise.
    singleNoiseLayer = np.zeros([gridSize+1, gridSize+1])

    # The underlying noise is the noise which is sampled at different frequencies in order to create the multi-layered noise.
    # The underlying noise is periodic.
    underlyingNoise = np.random.rand(gridSize+1, gridSize+1)
    underlyingNoise[:, -1] = underlyingNoise[:, 0]
    underlyingNoise[-1, :] = underlyingNoise[0, :]

    while samplingInterval >= 1:
        # Determines the rows and columns of the elements which are to be used for the current sampling frequency.
        sampleRows = range(0, gridSize+1, int(samplingInterval))
        sampleColumns = range(0, gridSize+1, int(samplingInterval))
        sampleRowGrid, sampleColumnGrid = np.meshgrid(sampleRows, sampleColumns)
        sampleRowList = np.reshape(sampleRowGrid, [sampleRowGrid.size, 1])
        sampleColumnList = np.reshape(sampleColumnGrid, [sampleColumnGrid.size, 1])

        # Extracts the random samples to be used to create the current layer.
        noiseSamples = underlyingNoise[sampleRowList, sampleColumnList]
        noiseSamples = np.reshape(noiseSamples, [int(np.sqrt(noiseSamples.size)), int(np.sqrt(noiseSamples.size))])

        # The sampled noise values are used to interpolate a whole layer that will be added to the multi-layered noise.
        # The RectBivariateSpline function can not handle input with less then/equal to 3 coordinates. When this is the
        # case the slower interp2d function is used instead, it uses linear interpolation.
        if len(sampleRows)>3:
            f = interpolate.RectBivariateSpline(sampleRows, sampleColumns, noiseSamples)
        else:
            f = interpolate.interp2d(sampleRowList, sampleColumnList, noiseSamples)
        layerRows = range(0, gridSize + 1, 1)
        layerColumns = range(0, gridSize + 1, 1)
        singleNoiseLayer = f(layerRows, layerColumns)

        generatedNoise += samplingAmplitude * singleNoiseLayer
        samplingInterval /= 2
        samplingAmplitude /= amplitudeReduction


    # The last row and column is removed to change the array to it's proper size. The noise is also normalized.
    generatedNoise = np.delete(generatedNoise, -1, axis=0)
    generatedNoise = np.delete(generatedNoise, -1, axis=1)
    generatedNoise -= np.min(generatedNoise)
    generatedNoise /= np.max(generatedNoise)
    return generatedNoise

def FractalNoise3D(gridSize, numberOfInitialIterationsToSkip = 1, amplitudeReduction = 2):

    samplingInterval = gridSize / pow(2, numberOfInitialIterationsToSkip)
    samplingAmplitude = 1 # This will decrease in each iteration.
    generatedNoise = np.zeros([gridSize+1, gridSize+1, gridSize+1])

    # The basic noise is sampled at different frequencies in order to create the final noise. The basic noise is periodic.
    basicNoise = np.random.rand(gridSize+1, gridSize+1, gridSize+1)
    basicNoise[-1, :, :] = basicNoise[0, :, :]
    basicNoise[:, -1, :] = basicNoise[:, 0, :]
    basicNoise[:, :, -1] = basicNoise[:, :, 0]

    #print(basicNoise)


    xFineMesh, yFineMesh, zFineMesh = np.meshgrid(range(0, gridSize+1), range(0, gridSize+1), range(0, gridSize+1))

    xFineVector = np.reshape(xFineMesh, ((gridSize + 1) ** 3, 1))
    yFineVector = np.reshape(yFineMesh, ((gridSize + 1) ** 3, 1))
    zFineVector = np.reshape(zFineMesh, ((gridSize + 1) ** 3, 1))

    #print(np.shape(np.transpose(xFineVector)))

    #finePoints = np.array([[np.transpose(xFineVector)], [np.transpose(yFineVector)], [np.transpose(zFineVector)]])
    #print(finePoints)
    #print(np.shape(finePoints))

    #np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)

    # finePoints is a n^3x3 array, each row corresponds to coordinates of the points in the basicNoise array.
    finePoints = np.append(xFineVector, yFineVector, axis = 1)
    finePoints = np.append(finePoints, zFineVector, axis = 1)

    #fig2 = mlab.figure()
    #fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    #mlab.points3d(finePoints[:, 0], finePoints[:, 1], finePoints[:, 2],
    #              basicNoise[finePoints[:, 0], finePoints[:, 1], finePoints[:, 2]], scale_factor=1.5, opacity=1)

    while samplingInterval >= 1:
        sampleX = range(0, gridSize + 1, int(samplingInterval))
        sampleY = range(0, gridSize + 1, int(samplingInterval))
        sampleZ = range(0, gridSize + 1, int(samplingInterval))
        sampleValues = basicNoise[0::int(samplingInterval), 0::int(samplingInterval), 0::int(samplingInterval)]

        print(np.shape(sampleValues))


        # The sampleX, sampleY and sampleZ values may be wrong.
        my_interpolating_function = interpolate.RegularGridInterpolator((sampleX, sampleY, sampleZ), sampleValues)
        interpolatedValues = my_interpolating_function(finePoints)
        #interpolatedValues = interpolate.interpn((sampleX, sampleY, sampleZ), sampleValues, finePoints)
        singleNoiseLayer = np.reshape(interpolatedValues, (gridSize+1, gridSize+1, gridSize+1))

        generatedNoise += samplingAmplitude * singleNoiseLayer
        samplingInterval /= 2
        samplingAmplitude /= amplitudeReduction

        #fig2 = mlab.figure()
        #fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        #mlab.points3d(finePoints[:, 0], finePoints[:, 1], finePoints[:, 2],
        #            singleNoiseLayer[finePoints[:, 0], finePoints[:, 1], finePoints[:, 2]], scale_factor=1.5, opacity = 1)

    #print(basicNoise[0:, 0:, 0:])
    #print('----------------------------------------')
    #print(basicNoise[0::2, 0::2, 0::2])
    #print('----------------------------------------')
    #print(basicNoise[0::4, 0::4, 0::4])

    generatedNoise -= np.min(generatedNoise)
    generatedNoise /= np.max(generatedNoise)

    #fig2 = mlab.figure()
    #fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    #mlab.points3d(finePoints[:, 0], finePoints[:, 1], finePoints[:, 2],
    #              generatedNoise[finePoints[:, 0], finePoints[:, 1], finePoints[:, 2]], scale_factor=1.5, opacity=1)
    #mlab.show()

    return finePoints, basicNoise, interpolatedValues, generatedNoise


def FractalNoiseSpherical(gridSize, sphericalPoints, numberOfInitialIterationsToSkip = 1, amplitudeReduction = 2):
    samplingInterval = gridSize / pow(2, numberOfInitialIterationsToSkip)
    samplingAmplitude = 1 # This will decrease in each iteration.
    generatedNoise = np.zeros((np.size(sphericalPoints, 0), 1))

    # The basic noise is sampled at different frequencies in order to create the final noise. The basic noise is periodic.
    basicNoise = np.random.rand(gridSize+1, gridSize+1, gridSize+1)
    basicNoise[-1, :, :] = basicNoise[0, :, :]
    basicNoise[:, -1, :] = basicNoise[:, 0, :]
    basicNoise[:, :, -1] = basicNoise[:, :, 0]

    while samplingInterval >= 1:
        sampleX = range(0, gridSize + 1, int(samplingInterval))
        sampleY = range(0, gridSize + 1, int(samplingInterval))
        sampleZ = range(0, gridSize + 1, int(samplingInterval))
        sampleValues = basicNoise[0::int(samplingInterval), 0::int(samplingInterval), 0::int(samplingInterval)]

        # The sampleX, sampleY and sampleZ values may be wrong.
        my_interpolating_function = interpolate.RegularGridInterpolator((sampleX, sampleY, sampleZ), sampleValues)
        sphereCoordinates = gridSize / 2 + 0.7 * gridSize * sphericalPoints / 2
        singleNoiseLayer = my_interpolating_function(sphereCoordinates)

        generatedNoise[:, 0] += samplingAmplitude * singleNoiseLayer
        samplingInterval /= 2
        samplingAmplitude /= amplitudeReduction
    generatedNoise -= np.min(generatedNoise)
    generatedNoise /= np.max(generatedNoise)

    return generatedNoise


def PerlinNoiseSpherical(gridSize, sphericalPoints, numberOfInitialIterationsToSkip = 1, amplitudeReduction = 2):


    # Indicates the location of the gradient vectors.
    xMesh, yMesh, zMesh = np.meshgrid(np.linspace(-1, 1, gridSize+1),
                                      np.linspace(-1, 1, gridSize+1),
                                      np.linspace(-1, 1, gridSize+1))
    xGradient = 2 * np.random.rand(gridSize + 1, gridSize + 1, gridSize + 1) - 1
    yGradient = 2 * np.random.rand(gridSize + 1, gridSize + 1, gridSize + 1) - 1
    zGradient = 2 * np.random.rand(gridSize + 1, gridSize + 1, gridSize + 1) - 1
    gradientMagnitude = np.sqrt(xGradient ** 2 + yGradient ** 2 + zGradient ** 2)
    xGradient /= gradientMagnitude
    yGradient /= gradientMagnitude
    zGradient /= gradientMagnitude
    xGradient[:, :, -1] = xGradient[:, :, 0]
    yGradient[:, :, -1] = yGradient[:, :, 0]
    zGradient[:, :, -1] = zGradient[:, :, 0]
    xGradient[:, -1, :] = xGradient[:, 0, :]
    yGradient[:, -1, :] = yGradient[:, 0, :]
    zGradient[:, -1, :] = zGradient[:, 0, :]
    xGradient[-1, :, :] = xGradient[0, :, :]
    yGradient[-1, :, :] = yGradient[0, :, :]
    zGradient[-1, :, :] = zGradient[0, :, :]

    meshDistance = xMesh[0, 1, 0] - xMesh[0, 0, 0]
    xMesh /= meshDistance
    yMesh /= meshDistance
    zMesh /= meshDistance
    sphericalPoints /= meshDistance
    minimumMeshValue = np.min(xMesh)
    xMesh -= minimumMeshValue
    yMesh -= minimumMeshValue
    zMesh -= minimumMeshValue
    sphericalPoints -= minimumMeshValue
    sphericalPoints *= 0.98
    sphericalPoints += 0.01 * sphericalPoints

    generatedNoise = np.zeros((np.size(sphericalPoints, 0), 1))

    basisFunctionDistance = np.zeros((8, 1))
    basisTemplate = np.array([[0, 0, 0],
                                          [1, 0, 0],
                                          [0, 1, 0],
                                          [0, 0, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [0, 1, 1],
                                          [1, 1, 1],
                                         ])

    for point, iPoint in zip(sphericalPoints, range(0, np.size(sphericalPoints, 0))):
        flooredPoint = np.floor(point)
        pointOffset = point - flooredPoint
        pointOffset = 6*pointOffset**5 - 15*pointOffset**4 + 10*pointOffset**3 # standard smooth interpolation.


        for i in range(8):
            basisFunctionDistance[i, 0] = np.sqrt((basisTemplate[i, 0]-pointOffset[0])**2
                                                  +(basisTemplate[i, 1]-pointOffset[1])**2
                                                  +(basisTemplate[i, 2]-pointOffset[2])**2)

        #pointOffset[:] /= np.sqrt(pointOffset[0]**2 + pointOffset[1]**2 + pointOffset[2]**2)
        #pointOffset[:] *= basisFunctionDistance[basisFunctionDistance == np.min(basisFunctionDistance)]
        #pointOffset[:] /= np.max(pointOffset)
        #print(basisFunctionDistance)
        #quit()
        #pointOffset = (np.sin(pointOffset*np.pi-np.pi/2)+1)/2
        # Meshes for the surrounding gradients coordinates.
        minorxMesh, minoryMesh, minorzMesh = np.meshgrid(
            np.linspace(flooredPoint[0], flooredPoint[0]+1, 2),
            np.linspace(flooredPoint[1], flooredPoint[1]+1, 2),
            np.linspace(flooredPoint[2], flooredPoint[2]+1, 2))

        minorxGradient = xGradient[minorxMesh.astype(int), minoryMesh.astype(int), minorzMesh.astype(int)]
        minoryGradient = yGradient[minorxMesh.astype(int), minoryMesh.astype(int), minorzMesh.astype(int)]
        minorzGradient = zGradient[minorxMesh.astype(int), minoryMesh.astype(int), minorzMesh.astype(int)]

        xDistanceVector = minorxMesh - point[0]
        yDistanceVector = minoryMesh - point[1]
        zDistanceVector = minorzMesh - point[2]
        # These 4 lines can be used to normalize the distance vectors.
        #r = np.sqrt(xDistanceVector**2 + yDistanceVector**2 + zDistanceVector**2)
        #xDistanceVector /= r
        #yDistanceVector /= r
        #zDistanceVector /= r

        influenceValues = minorxGradient*xDistanceVector + minoryGradient*yDistanceVector + minorzGradient*zDistanceVector

        # Performs the trilinear interpolation.
        influenceValues = influenceValues[0, :, :] * (1 - pointOffset[1]) + influenceValues[1, :, :] * pointOffset[1]
        influenceValues = influenceValues[0, :] * (1 - pointOffset[0]) + influenceValues[1, :] * pointOffset[0]
        #influenceValues = influenceValues[0, :, :] * (1 - pointOffset[0]) + influenceValues[1, :, :] * pointOffset[0]
        #influenceValues = influenceValues[0, :] * (1 - pointOffset[1]) + influenceValues[1, :] * pointOffset[1]
        influenceValues = influenceValues[0] * (1 - pointOffset[2]) + influenceValues[1] * pointOffset[2]

        # ---------------------------------------------
        # ---------------------------------------------
        generatedNoise[iPoint, 0] = influenceValues
        #generatedNoise[iPoint, 0] = np.sum(influenceValues) * basisFunctionDistance[basisFunctionDistance == np.min(basisFunctionDistance)]
        #generatedNoise[iPoint, 0] =  basisFunctionDistance[basisFunctionDistance == np.min(basisFunctionDistance)]
        #generatedNoise[iPoint, 0] = 2*np.sum(influenceValues) + np.sum(basisFunctionDistance)
        #print(basisFunctionDistance[basisFunctionDistance == np.min(basisFunctionDistance)])
        # ---------------------------------------------
        # ---------------------------------------------

    generatedNoise -= np.min(generatedNoise)
    generatedNoise /= np.max(generatedNoise)

    return generatedNoise