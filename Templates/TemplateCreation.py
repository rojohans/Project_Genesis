import Storage

import numpy as np
import pickle
from scipy.spatial import Delaunay


numberOfThetaValues = 300  # Used to generate the plate particles


thetaVector = np.linspace(-np.pi / 2, np.pi / 2, numberOfThetaValues)
dTheta = thetaVector[1] - thetaVector[0]
numberOfParticles = int(
    np.sum(np.fmax(np.round(np.cos(thetaVector) * 2 * np.pi / dTheta), np.ones(np.shape(thetaVector)))))
# particleCoordinates = [x, y, z, phi, theta, r]
particleCoordinates = np.zeros((numberOfParticles, 6))
iParticle = 0
for theta in thetaVector:
    phiVector = np.linspace(0, 2 * np.pi * (1 - 1 / np.max((np.round(np.cos(theta) * 2 * np.pi / dTheta), 1))),
                            np.max((np.round(np.cos(theta) * 2 * np.pi / dTheta), 1)))
    for phi in phiVector:
        r = 1
        particleCoordinates[iParticle, 0] = r * np.cos(phi) * np.cos(theta)
        particleCoordinates[iParticle, 1] = r * np.sin(phi) * np.cos(theta)
        particleCoordinates[iParticle, 2] = r * np.sin(theta)
        particleCoordinates[iParticle, 3] = phi
        particleCoordinates[iParticle, 4] = theta
        particleCoordinates[iParticle, 5] = r
        iParticle += 1

# incremental = True enables new points to be added, useful when the tectonic activity creates new land masses.
tri = Delaunay(particleCoordinates[:, 0:3])

# The calculated coordinates/simplices are stored in an object which will be stored to file.
worldTemplate = Storage.World3DTemplate(particleCoordinates[:, 0:3],
                        particleCoordinates[:, 3:6],
                        tri.simplices[:, 1:4],
                        numberOfParticles,
                        numberOfThetaValues)

# The template is stored in a .pkl file.
pickle.dump(worldTemplate, open('World_Template_' + str(numberOfThetaValues) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
#'Templates/' +

print('---DONE---')





