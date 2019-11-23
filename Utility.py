#
# This module contains functions which are not easily caterogicable .
#

import numpy as np

identityMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Used when rotating flow vectors.

def RotateVector(startPoint, endPoint, vectorToRotate):
    #
    # The function rotates the vector "vector" from "startPoint" to "endPoint". The vector is assumed to be
    # located at "startPoint" and after being rotated to "endPoint" is assumed to be located at that point. The vector at
    # "startPoint" and "endPoint" will be the same from the perspective of the "startPoint" and "endPoint" respectively.
    #

    # The crossVector specifies the axes around which the rotation will occur.
    crossVector = np.cross(startPoint, endPoint)

    skewMatrix = np.array([[0, -crossVector[2], crossVector[1]],
                           [crossVector[2], 0, -crossVector[0]],
                           [-crossVector[1], crossVector[0], 0]])
    skewMatrixSquared = np.dot(skewMatrix, skewMatrix)

    s = np.sqrt(crossVector[0] ** 2 + crossVector[1] ** 2 + crossVector[2] ** 2)
    c = np.dot(endPoint, startPoint)
    R = identityMatrix + skewMatrix + skewMatrixSquared * (1 - c) / (s ** 2 + 0.000001)

    vectorToRotate = startPoint + vectorToRotate
    vLength = np.sqrt(vectorToRotate[0] ** 2 + vectorToRotate[1] ** 2 + vectorToRotate[2] ** 2)
    vectorToRotate /= vLength  # scales to 1

    rotatedVector = np.dot(R, vectorToRotate)

    rotatedVector *= vLength
    rotatedVector -= endPoint
    return rotatedVector

def RotateVector2Steps(startPoint, endPoint, vectorToRotate):
    rhoCandidate = np.sqrt(startPoint[0] ** 2 + startPoint[1] ** 2)
    rhoMain = np.sqrt(endPoint[0] ** 2 + endPoint[1] ** 2)
    if rhoCandidate == 0:
        midpoint = np.array([0, 0, endPoint[2]])
    else:
        midPoint = np.array(
            [startPoint[0] * rhoMain / rhoCandidate, startPoint[1] * rhoMain / rhoCandidate,
             endPoint[2]])

    # Theta rotation
    rotatedVector = RotateVector(startPoint=startPoint,
                                         endPoint=midPoint,
                                         vectorToRotate=vectorToRotate)
    # Phi rotation
    rotatedVector = RotateVector(startPoint=midPoint,
                                         endPoint=endPoint,
                                         vectorToRotate=rotatedVector)
    return rotatedVector

















# calculating phi/theta from [x, y, z] variables.
'''
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

dTheta = mainTheta - candidateTheta
dPhi = mainPhi - candidatePhi
'''