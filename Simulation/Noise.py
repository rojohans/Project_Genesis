import numpy as np
from scipy import interpolate


def SimpleNoise(gridSize, numberOfInitialIterationsToSkip = 1, amplitudeReduction = 2):
    '''
    # --------ALGORITHM SUMMARY---------
    # An underlying noise array is created with values uniformly distributed between 0 and 1. The underlying noise is
    # sampled at different frequencies and interpolated to give rise to different layers of noise. Each layer is an
    # interpretation of the underlying noise at a certain resolution. The layers are weighted according to their
    # resolution and then combined to form a final multi-layered noise array. The multi-layered noise contain
    # information from all resolutions in varying amount depending on the weights. The noise is naturally periodic.
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

    # The underlying noise is the noise wihich is sampled at different frequencies in order to create the multi-layered noise.
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
