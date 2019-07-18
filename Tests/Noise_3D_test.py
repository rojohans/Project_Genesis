import numpy as np
import Simulation.Templates

import Simulation.Noise

from scipy import interpolate
from scipy.spatial import SphericalVoronoi
from scipy.spatial import Delaunay

import Visualization
import mayavi.mlab

from mayavi.mlab import *
from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor

import pickle



if False:
    '''
    a = Simulation.Noise.SimpleNoise(512, 1, 2)
    print(a)

    b = np.matlib.repmat(a, 2, 2)

    fig1 = mayavi.mlab.figure()
    fig1.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    mayavi.mlab.surf(200*b)
    mayavi.mlab.show()
    '''
else:
    #                                                                        gridSize, variability, roughness/smoothness
    #finePoints, basicNoise, interpolatedValues, generatedNoise = Simulation.Noise.FractalNoise3D(gridSize, 2, 2)

    '''
    connections = ((0, 2), (3, 5))  # point 0 and 2 and 3 and 5 are connected
    x = np.random.randn(10)
    y = np.random.randn(10)
    z = np.random.randn(10)
    pts = mlab.points3d(x, y, z)

    pts.mlab_source.dataset.lines = np.array(connections)

    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    tube.filter.radius_factor = 1.
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0))

    print("visualization done")
    mlab.show()


    #mlab.test_molecule()
    mlab.show()
    quit()
    '''

    #gridSize = 256
    numberOfThetaValues = 300  # Used as the 2D resolution (# theta values, the # phi values is twice this number.)
    faceColorSmoothing = False   # False: Each face will have a single colour based of the corner values.
                                 #  True: The face colour varies throughout the face, interpolated from the corner values.
    projectTopography = True     # False: The heightvalues of the points will be used to illustrate height.
                                 #  True: The surface is projected on a spherical shell.
    fractalNoise = True # Determines wether a combination of noises should be used, or juts one.
    CalculateFlow = False # Determines if the vertex noise should be converted into a vector field.

    # For division values of 8 or greater the visualization becomes slow and laggy. (On mac)
    world = Simulation.Templates.IcoSphere(6)
    vertsArray = np.asarray(world.vertices)
    facesArray = np.asarray(world.faces)

    print('#vertices = ', np.size(vertsArray, 0))
    print('#faces', np.size(facesArray, 0))
    print('Subdivision done')

    b = 1.75 # Used as fractal multiplier.

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
        interpolatedRadius = Simulation.Noise.PerlinNoiseSpherical(32,
                                                                    vertsArray.copy(),
                                                                    3,
                                                                    1.5)
    interpolatedRadius -= np.min(interpolatedRadius)
    interpolatedRadius /= np.max(interpolatedRadius)

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
    #for cdf in cdfValues:
        #interpolatedRadius[interpolatedRadius == np.min(interpolatedRadius)] = cdf
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
    mlab.plot3d(np.linspace(0, 1, N), constantY[:, 0], (np.sin(x*np.pi-np.pi/2)+1)/2, tube_radius = None, line_width = 0.5)
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
        vertexFlowAngle = interpolatedRadius*2*np.pi
        vertexXFlow = - np.cos(vertexFlowAngle[:, 0]) * np.sin(vertexPhiAngle)\
                      + np.sin(vertexFlowAngle[:, 0]) * np.cos(vertexPhiAngle)*np.sin(vertexThetaAngle)
        vertexYFlow = np.cos(vertexFlowAngle[:, 0]) * np.cos(vertexPhiAngle)\
                      + np.sin(vertexFlowAngle[:, 0]) * np.sin(vertexPhiAngle)*np.sin(vertexThetaAngle)
        vertexZFlow = np.sin(vertexFlowAngle[:, 0]) * np.cos(vertexThetaAngle)
        print('Vector field generated')
        #----------------------------------------------------------------





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


        numberOfFlowParticles = 5
        numberOfSteps = 100
        #deltaAngle = np.pi/10
        deltaVelocity = np.pi/200
        maxVelocity = 0.02

        u = np.transpose(np.random.rand(numberOfFlowParticles, 1))
        v = np.transpose(np.random.rand(numberOfFlowParticles, 1))
        flowParticlePhiAngle = 2 * np.pi * u
        flowParticleThetaAngle= np.arccos((2 * v - 1)) - np.pi / 2

        # Stores the trajectory of each particle. [#steps, #particles]
        flowParticlePhiTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
        flowParticlePhiVelocityTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
        flowParticleThetaTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))
        flowParticleThetaVelocityTrajectory = np.zeros((numberOfSteps, numberOfFlowParticles))

        # Phi and theta values of the flow vectors.
        phiArray = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
        thetaArray = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))


        trajectoryDensity = np.zeros((np.size(interpolatedRadius, 0), 1)) # Used for visualizing the trejectory density.

        for phi, theta, iParticle in zip(np.transpose(flowParticlePhiAngle), np.transpose(flowParticleThetaAngle), range(numberOfFlowParticles)):
            phiVelocity = 0
            thetaVelocity = 0

            flowParticlePhiTrajectory[0, iParticle] = phi
            flowParticlePhiVelocityTrajectory[0, iParticle] = 0
            flowParticleThetaTrajectory[0, iParticle] = theta
            flowParticleThetaVelocityTrajectory[0, iParticle] = 0
            for iStep in range(1, numberOfSteps):
                arcDistance = np.arccos(np.cos(phi)* np.cos(theta) * vertsArray[:, 0]
                                        + np.sin(phi) * np.cos(theta) * vertsArray[:, 1]
                                        + np.sin(theta) * vertsArray[:, 2])

                trajectoryDensity[arcDistance == np.min(arcDistance)] += 1
                localFlowAngle = vertexFlowAngle[arcDistance == np.min(arcDistance)]

                phiVelocity = (2*phiVelocity + np.cos(localFlowAngle[:, 0])*maxVelocity)/3
                #phiVelocity = np.cos(localFlowAngle)/50
                thetaVelocity = (2*thetaVelocity + np.sin(localFlowAngle[:, 0])*maxVelocity)/3
                #thetaVelocity = np.sin(localFlowAngle)/50

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

        mlab.quiver3d(vertsArray[:, 0],
                      vertsArray[:, 1],
                      vertsArray[:, 2],
                      vertexXFlow,
                      vertexYFlow,
                      vertexZFlow,
                      scale_factor = 0.015,
                      mode = 'cone') # 'arrow'
        heightSurface = mlab.triangular_mesh(
            vertsArray[:, 0]*0.99,
            vertsArray[:, 1]*0.99,
            vertsArray[:, 2]*0.99,
            facesArray,
            scalars=interpolatedRadius[:, 0]*0,
            colormap='gist_earth')

        print('Visualization done')



        #==========================================
        # Visualizes the trajectory density over the sphere in the 2D plane.


        phiMesh, thetaMesh = np.meshgrid(
            np.linspace(0, 2*np.pi, 2*numberOfThetaValues),
            np.linspace(-np.pi/2, np.pi/2, numberOfThetaValues))
        phiMesh = np.transpose(phiMesh)
        thetaMesh = np.transpose(thetaMesh)

        phiArray = np.arctan2(vertsArray[:, 1], vertsArray[:, 0]) + np.pi
        thetaArray = np.arctan2(vertsArray[:, 2], np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2))
        r = np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2 + vertsArray[:, 2]**2)

        #interpolatedRadius
        #radiusMesh = interpolate.griddata((phiArray, thetaArray), particleID, (phiMesh, thetaMesh))
        radiusMesh = interpolate.griddata((phiArray, thetaArray), trajectoryDensity, (phiMesh, thetaMesh))
        radiusMesh = radiusMesh[:, :, 0]
        #vertsArray[:, 0:2]

        fig3 = mlab.figure()
        fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


        mlab.mesh(phiMesh, thetaMesh, 0*radiusMesh, scalars = radiusMesh, colormap = 'gist_earth')
        # ==========================================



        mlab.show()

        quit()
    # ------------------------------------------------------------------------




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

    # The radiuses are adjusted.
    radiusModifier = 15.8
    vertexRadius = radiusModifier + interpolatedRadius + 0*crustThickness
    vertsArray[:, 0] *= vertexRadius[:, 0]
    vertsArray[:, 1] *= vertexRadius[:, 0]
    vertsArray[:, 2] *= vertexRadius[:, 0]

    fig1 = mlab.figure()
    fig1.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    #mlab.points3d(hotSpotCoordinates[:, 0]*1.5, hotSpotCoordinates[:, 1]*1.5, hotSpotCoordinates[:, 2]*1.5,
    #              color=(1, 1, 1), scale_factor = 0.1)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
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
    r = np.sqrt(vertsArray[:, 0]**2 + vertsArray[:, 1]**2 + vertsArray[:, 2]**2)

    #interpolatedRadius
    #radiusMesh = interpolate.griddata((phiArray, thetaArray), particleID, (phiMesh, thetaMesh))
    radiusMesh = interpolate.griddata((phiArray, thetaArray), vertexRadius, (phiMesh, thetaMesh))
    radiusMesh = radiusMesh[:, :, 0]
    #vertsArray[:, 0:2]

    fig3 = mlab.figure()
    fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()


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
                r = 1
                particleCoordinates[iParticle, 0] = r * np.cos(phi) * np.cos(theta)
                particleCoordinates[iParticle, 1] = r * np.sin(phi) * np.cos(theta)
                particleCoordinates[iParticle, 2] = r * np.sin(theta)
                particleCoordinates[iParticle, 3] = r
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







