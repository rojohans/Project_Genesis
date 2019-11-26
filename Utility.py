#
# This module contains functions which are not easily caterogicable .
#

import numpy as np

identityMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Used when rotating flow vectors.

def RotateVector(startPoint, endPoint, vectorToRotate, a = 0):
    #
    # The function rotates the vector "vector" from "startPoint" to "endPoint". The vector is assumed to be
    # located at "startPoint" and after being rotated to "endPoint" is assumed to be located at that point. The vector at
    # "startPoint" and "endPoint" will be the same from the perspective of the "startPoint" and "endPoint" respectively.
    #

    # The crossVector specifies the axes around which the rotation will occur.

    if a == 1:
        v1 = startPoint.copy()
        v2 = endPoint.copy()
        v1[2] = 0
        v2[2] = 0
        crossVector = np.cross(v1, v2)
    else:
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
        midPoint = np.array([0, 0, endPoint[2]])
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
                                         vectorToRotate=rotatedVector,
                                         a = 1)
    return rotatedVector

def VectorDistance(v1, v2):
    # Return the 2norm of the difference between two vectors.
    return np.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2 + (v1[2]-v2[2])**2 )

def CaartesianToSpherical(point):
    theta = np.arcsin(point[2])
    if point[0]==0:
        # Take care of /0 case. The phi angle should be -pi/2 or pi/2 depending on the y-value.
        if point[1]>0:
            phi = np.pi/2
        else:
            phi = -np.pi/2
    else:
        phi = np.arctan(point[1] / point[0]) + np.pi * (1 - np.sign(point[0])) / 2
    radius = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    return phi, theta, radius











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