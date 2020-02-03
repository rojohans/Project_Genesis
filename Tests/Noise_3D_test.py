import numpy as np
import Templates.Templates

import Simulation.Noise
import Simulation.PlateDynamics

import Visualization
import Utility

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
tic = time.perf_counter()
#world = Simulation.Templates.IcoSphere(6)

world = Templates.Templates.GetIcoSphere(6) # use 4 when testing tectonic plates.
# IcoSphereSimple creates an icosphere without the neighbour lists.
#world = Templates.Templates.IcoSphereSimple(6)


# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
# Consider saving the IcoSphere data to a .csv file instead of a .pkl file. This could improve load times.
# The Neighbour object within the ico sphere takes up considerable amounts of memory. Consider just calculating the
# neighbours for one distance value, instead of a range.
# Consider saving the ico sphere as multiple .csv files.
# 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
toc = time.perf_counter()
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
    # Tests to rotate all vectors to a given point. The vectors the the phi or theta unit vectors.
    phiVector = np.zeros((world.numberOfvertices, 1))
    thetaVector = np.zeros((world.numberOfvertices, 1))

    xFlow = np.zeros((world.numberOfvertices, 1))
    yFlow = np.zeros((world.numberOfvertices, 1))
    zFlow = np.zeros((world.numberOfvertices, 1))


    for iPoint, point in enumerate(world.vertices):
        '''
        thetaVector[iPoint] = np.arcsin(point[2])
        if point[0]==0:
            # Take care of /0 case. The phi angle should be -pi/2 or pi/2 depending on the y-value.
            if point[1]>0:
                phiVector[iPoint] = np.pi/2
            else:
                phiVector[iPoint] = -np.pi/2
        else:
            phiVector[iPoint] = np.arctan(point[1] / point[0]) + np.pi * (
                1 - np.sign(point[0])) / 2
        '''
        phiVector[iPoint], thetaVector[iPoint], radius = Utility.CaartesianToSpherical(point)
        # Phi unit vectors.
        xFlow[iPoint] = -np.sin(phiVector[iPoint])
        yFlow[iPoint] = np.cos(phiVector[iPoint])
        zFlow[iPoint] = 0
        # Theta unit vector
        #xFlow[iPoint] = -np.cos(phiVector[iPoint])*np.sin(thetaVector[iPoint])
        #yFlow[iPoint] = -np.sin(phiVector[iPoint])*np.sin(thetaVector[iPoint])
        #zFlow[iPoint] = np.cos(thetaVector[iPoint])

    flowVectors = np.append(xFlow, yFlow, axis = 1)
    flowVectors = np.append(flowVectors, zFlow, axis = 1)

    flowVectorsRotated = flowVectors.copy()
    fixedPoint = [1/np.sqrt(2), 0, 1/np.sqrt(2)]#[1/np.sqrt(2), 1/(np.sqrt(2)), 0]
    #fixedPoint = [1, 0, 0]
    tree = scipy.spatial.cKDTree(world.vertices)
    q = tree.query(fixedPoint)
    fixedPoint = world.vertices[q[1], :]
    fixedFlow = flowVectors[q[1], :]
    error = np.zeros((world.numberOfvertices, 1))

    print(np.sqrt(fixedPoint[0]**2 + fixedPoint[1]**2 + fixedPoint[2]**2))
    for iVertex in range(world.numberOfvertices):
        tmp = Utility.RotateVector2Steps(world.vertices[iVertex, :].copy(), fixedPoint, flowVectors[iVertex, :].copy())
        flowVectorsRotated[iVertex, :] = tmp
        error[iVertex] = Utility.VectorDistance(fixedFlow, tmp)


    Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                                 faces = world.faces.copy(),
                                 radius = world.radius.copy(),
                                 scalars = error,
                                 projectTopography = True,
                                 projectRadiusSpan = [1, 1.03],
                                 interpolatedTriangleColor = True,
                                 colormap = 'gist_earth',
                                 randomColormap = False)

    Visualization.VisualizeFlow(world.vertices,
                                flowVectorsRotated[:, 0],
                                flowVectorsRotated[:, 1],
                                flowVectorsRotated[:, 2],
                                world.faces,
                                newFigure = True,
                                sizeFactor = 0.03)
    print('Visualization done')
    mlab.show()
    quit()


if True:
    import Root_Directory
    import System_Info

    saveScreenshots = True

    fileName = Root_Directory.Path() + '/Templates/Plate_Collection_' + str(1.2) + '.pkl'
    #fileName = Root_Directory.Path() + '/Templates/Plate_Collection_1_2' + '.pkl'
    fileToRead = open(fileName, 'rb')
    plateCollection = pickle.load(fileToRead)
    fileToRead.close()

    plateDictionary = plateCollection.plateList

    world.kdTree = scipy.spatial.cKDTree(world.vertices)

    xFlow = np.reshape(plateCollection.xFlow, (world.numberOfvertices, 1))
    yFlow = np.reshape(plateCollection.yFlow, (world.numberOfvertices, 1))
    zFlow = np.reshape(plateCollection.zFlow, (world.numberOfvertices, 1))
    flowVectors = np.append(xFlow, yFlow, axis=1)
    flowVectors = np.append(flowVectors, zFlow, axis=1)
    # These are the rotations exises for each vertex.
    flowRotationVectors = np.zeros((world.numberOfvertices, 3))
    for iVertex, flow in enumerate(flowVectors):
        flowRotationVectors[iVertex, :] = np.cross(world.vertices[iVertex, :], flow)

    
    Simulation.PlateDynamics.Plate.LinkLists(plateDictionary,
                                             world.kdTree,
                                             flowRotationVectors,
                                             world)
    Simulation.PlateDynamics.Plate.UpdateKDTree()
    for key, plate in plateDictionary.items():
        plate.MeshTriangulation()
        plate.UpdateNearestPointIndex()
        plate.UpdateFlow()
        plate.UpdateAverage()
        plate.CalculateAdjacentPlates()
    #    print(plate.averageFlowVector)
    #    print('----------')
    #Simulation.PlateDynamics.Plate.CheckForMerge()


    #for key, plate in plateDictionary.items():
    #    for iPoint in range(plate.numberOfVertices):
    #
    #        plate.vertices[iPoint, :] = Utility.RotateAroundAxis(plate.vertices[iPoint, :], plate.averageFlowVector, 1*np.pi/180)
    #    break

    world4 = Templates.Templates.IcoSphereSimple(4)
    treeResult = world.kdTree.query(world4.vertices, 1)

    '''
    import time
    import numpy as np
    from mayavi import mlab

    f = mlab.figure()
    V = np.random.randn(20, 20, 20)
    s = mlab.contour3d(V, contours=[0])

    @mlab.animate(delay=10, ui = False)
    def anim():
        i = 0
        while i < 5:
            time.sleep(1)
            s.mlab_source.set(scalars=np.random.randn(20, 20, 20))
            i += 1
            yield
    anim()
    mlab.show()
    quit()
    '''

    import wx

    app = wx.App()
    frame = wx.Frame(parent=None, title='Hello World')
    frame.Show()
    app.MainLoop()
    quit()



    from numpy import ogrid, sin

    from traits.api import HasTraits, Instance
    from traitsui.api import View, Item

    from mayavi.sources.api import ArraySource
    from mayavi.modules.api import IsoSurface

    from mayavi.core.ui.api import SceneEditor, MlabSceneModel


    class MayaviView(HasTraits):

        scene = Instance(MlabSceneModel, ())

        # The layout of the panel created by Traits
        view = View(Item('scene', editor=SceneEditor(), resizable=True,
                         show_label=False),
                    resizable=True)

        def __init__(self):
            HasTraits.__init__(self)
            # Create some data, and plot it using the embedded scene's engine
            x, y, z = ogrid[-10:10:100j, -10:10:100j, -10:10:100j]
            scalars = sin(x * y * z) / (x * y * z)
            src = ArraySource(scalar_data=scalars)
            self.scene.engine.add_source(src)
            src.add_module(IsoSurface())


    # -----------------------------------------------------------------------------
    # Wx Code
    import wx


    class MainWindow(wx.Frame):

        def __init__(self, parent, id):
            wx.Frame.__init__(self, parent, id, 'Mayavi in Wx')
            self.mayavi_view = MayaviView()
            # Use traits to create a panel, and use it as the content of this
            # wx frame.
            self.control = self.mayavi_view.edit_traits(
                parent=self,
                kind='subpanel').control
            self.Show(True)


    app = wx.PySimpleApp()
    frame = MainWindow(None, wx.ID_ANY)
    app.MainLoop()


    quit()




    import time
    import numpy as np
    from mayavi import mlab
    import wx

    V = np.random.randn(20, 20, 20)
    f = mlab.figure()
    s = mlab.contour3d(V, contours=[0])


    def animate_sleep(x):
        n_steps = int(x / 0.01)
        for i in range(n_steps):
            time.sleep(0.01)
            #help(wx.Yield)
            wx.YieldIfNeeded()

    for i in range(5):
        animate_sleep(1)

        V = np.random.randn(20, 20, 20)

        # Update the plot with the new information
        s.mlab_source.set(scalars=V)
    quit()



    for key in plateDictionary:
        plateDictionary[key].UpdateSurfaceKDTree()

    # Creates a GUI.
    if True:
        tic = time.perf_counter()
        for key in plateDictionary:
            plate = plateDictionary[key]
            if plate.numberOfVertices > 4:
                plate.FindBorderPoints()
                plate.ConnectBorderPoints(sort = True)
                plate.UpdateBorderKDTree()

                plate.FindSecondBorderPoints()

                #plate.UpdateSurfaceKDTree()
                #plate.ConnectEdgePoints(sort = False)
        toc = time.perf_counter()
        print('plate border calculating/sorting : ', toc-tic)


        tic = time.perf_counter()
        for key in plateDictionary:
            plate = plateDictionary[key]
            if plate.numberOfVertices > 4:
                #plate.CalculateAdjacentPlates()
                #print(plate.adjacentPlateIDs)
                plate.CalculateInteractionStress()
        toc = time.perf_counter()
        print('Interaction stress calculated : ', toc - tic)


        nPoints = 0
        iPlate = -1
        borderIndex = 0
        for key, plate in plateDictionary.items():
            iPlate += 1
            if iPlate < 100:
                if plate.numberOfVertices > 4:
                    # plateVerts.append(plate.vertices)
                    nBorderPoints = np.size(plate.borderVertex, 0)
                    if iPlate == 0:
                        plateVerts = plate.vertices
                        plateFaces = plate.triangles
                        meshScalarsID = plate.ID * np.ones((plate.numberOfVertices, 1))
                        meshScalarsNumber = iPlate * np.ones((plate.numberOfVertices, 1))
                        scalarStress = plate.stressVector
                        scalarInteractionStress = plate.interactionStress
                        # meshScalarsNumber = plate.ID * np.ones((plate.numberOfVertices, 1))
                        borderVertices = plate.borderVertex
                        borderLines = plate.borderLines
                        borderScalars = iPlate * np.ones((nBorderPoints, 1))
                        secondBorderScalars = iPlate * np.ones(( np.size(plate.secondBorderIndices, 0) , 1))
                        secondBorderVertices = plate.secondBorderPoints
                    else:
                        plateVerts = np.append(plateVerts, plate.vertices, axis=0)
                        plateFaces = np.append(plateFaces, plate.triangles + nPoints, axis=0)
                        meshScalarsID = np.append(meshScalarsID, plate.ID * np.ones((plate.numberOfVertices, 1)), axis=0)
                        meshScalarsNumber = np.append(meshScalarsNumber, iPlate * np.ones((plate.numberOfVertices, 1)), axis=0)
                        scalarStress = np.append(scalarStress, plate.stressVector, axis=0)
                        scalarInteractionStress = np.append(scalarInteractionStress, plate.interactionStress, axis=0)


                        borderVertices = np.append(borderVertices, plate.borderVertex, axis=0)
                        borderLines = np.append(borderLines, borderIndex + plate.borderLines, axis = 0)
                        borderScalars = np.append(borderScalars, iPlate * np.ones((nBorderPoints, 1)),axis=0)
                        secondBorderScalars = np.append(secondBorderScalars,
                                                        iPlate * np.ones(( np.size(plate.secondBorderIndices, 0) , 1)),
                                                        axis=0)
                        secondBorderVertices = np.append(secondBorderVertices, plate.secondBorderPoints, axis=0)
                    nPoints += plate.numberOfVertices
                    borderIndex += nBorderPoints
            else:
                break

        # Creates/retrieves the colormaps to be used.
        colormapRandom = np.random.randint(0, 255, (256, 3))
        import Templates.Colormap.Colormap as Colormap
        colormapHeat = Colormap.LoadColormap('gist_heat')

        # Rescales scalar values to fit the colormaps.
        meshScalarsNumber /= np.max(meshScalarsNumber)
        meshScalarsNumber *= 255
        meshScalarsNumber = np.round(meshScalarsNumber)
        print(np.min(scalarStress))
        print(np.max(scalarStress))
        scalarStress /= 2
        #scalarStress *= 255/2
        #scalarStrees = np.round(scalarStress)
        borderScalars /= np.max(borderScalars)
        borderScalars *= 255
        borderScalars = np.round(borderScalars)

        secondBorderScalars /= np.max(secondBorderScalars)
        secondBorderScalars *= 255
        secondBorderScalars = np.round(secondBorderScalars)

        scalarInteractionStress /= np.max(scalarInteractionStress)
        print(np.min(scalarStress))
        print(np.max(scalarStress))




        '''
        # Create a new mayavi scene.
        mayavi.new_scene()

        # Get the current active scene.
        s = mayavi.engine.current_scene

        # Read a data file.
        d = mayavi.open('fire_ug.vtu')

        # Import a few modules.
        from mayavi.modules.api import Outline, IsoSurface, Streamline

        # Show an outline.
        o = Outline()
        mayavi.add_module(o)
        o.actor.property.color = 1, 0, 0  # red color.

        # Make a few contours.
        iso = IsoSurface()
        mayavi.add_module(iso)
        iso.contour.contours = [450, 570]
        # Make them translucent.
        iso.actor.property.opacity = 0.4
        # Show the scalar bar (legend).
        iso.module_manager.scalar_lut_manager.show_scalar_bar = True

        # A streamline.
        st = Streamline()
        mayavi.add_module(st)
        # Position the seed center.
        st.seed.widget.center = 3.5, 0.625, 1.25
        st.streamline_type = 'tube'

        # Save the resulting image to a PNG file.
        s.scene.save('test.png')

        # Make an animation:
        for i in range(36):
            # Rotate the camera by 10 degrees.
            s.scene.camera.azimuth(10)

            # Resets the camera clipping plane so everything fits and then
            # renders.
            s.scene.reset_zoom()

            # Save the scene.
            s.scene.save_png('anim%d.png' % i)
        mlab.show()
        quit()
        '''


        from numpy import linspace, pi, cos, sin
        from traits.api import HasTraits, Range, Instance, on_trait_change, Button, Enum
        from traitsui.api import View, Item, HGroup
        from mayavi.tools.mlab_scene_model import MlabSceneModel
        from tvtk.pyface.scene_editor import SceneEditor

        class MayaviWindow(HasTraits):
            #n_turns = Range(0, 30, 11)
            flow_toggle = Button(label = 'Toggle Flow On/Off')
            randomize_plate_colours = Button()
            quit_button = Button(label = 'QUIT')
            simulate_button = Button(label = 'SIMULATE')
            surface_scalar_mode = Enum('ID', 'Interaction_Stress', 'Stress')
            mayaviScene = Instance(MlabSceneModel, ())
            #scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

            flowVisibility = 1.0
            a = 0

            def __init__(self, colormapRandom, colormapHeat):
                HasTraits.__init__(self)
                self.colormapRandom = colormapRandom
                self.colormapHeat = colormapHeat

                self.visObjBlack = Visualization.VisualizeGlobe(vertices=0.98 * world4.vertices.copy(),
                                                           faces=world4.faces.copy(),
                                                           radius=world4.radius.copy(),
                                                           scalars=world4.radius.copy(),
                                                           projectTopography=True,
                                                           projectRadiusSpan=[1, 1.03],
                                                           interpolatedTriangleColor=True,
                                                           colormap='gist_earth',
                                                           figure=self.mayaviScene)
                self.borderObj = Visualization.VisualizeTubes(borderVertices, borderLines, borderScalars, colormapRandom, scene = self.mayaviScene)

                self.surfaceObj = Visualization.VisualizeGlobe(vertices=plateVerts,
                                                      faces=plateFaces,
                                                      radius=np.ones((nPoints, 1)),
                                                      scalars=meshScalarsNumber,
                                                      projectTopography=True,
                                                      projectRadiusSpan=[1, 1.03],
                                                      interpolatedTriangleColor=True,
                                                      customColormap = colormapRandom,
                                                      figure=self.mayaviScene)


                self.flowObj = Visualization.VisualizeFlow(1.02 * world.vertices,
                                                      plateCollection.xFlow,
                                                      plateCollection.yFlow,
                                                      plateCollection.zFlow,
                                                      world.faces,
                                                      newFigure=False,
                                                      arrowColor=(1, 1, 1),
                                                      sizeFactor=0.02)
                '''
                self.flowObj = Visualization.VisualizeFlow(1.02 * world4.vertices,
                                                      plateCollection.xFlow[treeResult[1]],
                                                      plateCollection.yFlow[treeResult[1]],
                                                      plateCollection.zFlow[treeResult[1]],
                                                      world4.faces,
                                                      newFigure=False,
                                                      arrowColor=(1, 1, 1),
                                                      sizeFactor=0.03)
                '''

                '''
                # Displays the "second border points", these are the points adjacent to the borderpoints.
                self.secondBorder = self.mayaviScene.mlab.points3d(secondBorderVertices[:, 0],
                                                                   secondBorderVertices[:, 1],
                                                                   secondBorderVertices[:, 2],
                                                                   scale_factor = 0.005)
                self.secondBorder.glyph.scale_mode = 'scale_by_vector'
                # The scalar needs to be in the range [0, 1]
                self.secondBorder.mlab_source.dataset.point_data.scalars = secondBorderScalars[:, 0] / 255
                self.secondBorder.mlab_source.dataset.point_data.scalars.name = 'secondBorderScalars'
                lut = self.secondBorder.module_manager.scalar_lut_manager.lut.table.to_array()
                lut[:, 0:3] = colormapRandom
                self.secondBorder.module_manager.scalar_lut_manager.lut.table = lut
                '''

            def your_function(self, obj, event):
                #help(obj)
                #print(obj)
                print(obj.GetKeyCode())
                self._flow_toggle_fired()
                #GetEventPosition(...)
                #GetMousePosition(...)
                #GetPicker(...)
                #StartPickCallback(...)
                 #|      V.StartPickCallback()
                 #|      C++: virtual void StartPickCallback()
                 #|
                 #|      These methods correspond to the Exit, User and Pick callbacks.
                 #|      They allow for the Style to invoke them.

            @on_trait_change('mayaviScene.activated')
            def InitialSceneSettings(self):
                '''
                Changes the interactor. The mlab.view() call is neccesary for the interactor to be aligned verticaly with teh z-axis.
                :return:
                '''
                self.mayaviScene.mlab.view(0, 90)
                self.mayaviScene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

                '''
                # This code is supposed to create an override for the standard hot keys, but it doesn't work.
                class MyInteractorStyle(tvtk.InteractorStyleUser):

                    def __init__(self, parent=None):
                        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
                        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
                        #self.AddObserver("KeyPressEvent", self.key_press_event)

                    def middle_button_press_event(self, obj, event):
                        print("Middle Button pressed")
                        tvtk.InteractorStyleTerrain.OnMiddleButtonDown()
                        return
                    #def key_press_event(selfobj, obj, event):
                    #    print('Howdy')
                    #    return

                    def middle_button_release_event(self, obj, event):
                        print("Middle Button released")
                        tvtk.InteractorStyleTerrain.OnMiddleButtonUp()
                        return
                self.mayaviScene.interactor.interactor_style = MyInteractorStyle()
                
                #self.mayaviScene.interactor.add_observer('KeyPressEvent', self.your_function, 9999.0)
                #self.mayaviScene.interactor.add_observer('MouseMoveEvent', mouse_move_callback)
                '''

                # The scalarbar must be created after the interactor is created, thusly it cannot be created in the
                # constructor like all the other mayavi visualization objects.
                # mayavi.mlab.scalarbar and mayavi.mlab.colorbar seems to be the same.
                self.scalarBar = self.mayaviScene.mlab.scalarbar(self.surfaceObj.mayaviMeshObject,
                                                                 orientation = 'vertical',
                                                                 nb_labels = 6)

            @on_trait_change('surface_scalar_mode')
            def ChangeSurfaceScalarMode(self):
                '''
                Changes the scalars used to sample the colormap.
                This will not work if interpolatedTriangleColor = False when Visualization.VisualizeGlobe is called. For that
                one has to add alternative code.
                :return:
                '''
                #self.mayaviScene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
                print(self.surface_scalar_mode)
                if self.surface_scalar_mode is 'ID':
                    self.surfaceObj.mayaviMeshObject.mlab_source.set(scalars = meshScalarsNumber[:, 0])

                    #lut = self.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
                    #lut[:, 0:3] = customColormap
                    #print(np.shape(lut))
                    #self.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut
                    lut = self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
                    lut[:, 0:3] = self.colormapRandom
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.data_range = (0, 255)

                elif self.surface_scalar_mode is 'Interaction_Stress':
                    self.surfaceObj.mayaviMeshObject.mlab_source.set(scalars=scalarInteractionStress[:, 0])
                    lut = self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
                    lut[:, 0:3] = self.colormapHeat
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.data_range = (0, 1)

                elif self.surface_scalar_mode is 'Stress':
                    self.surfaceObj.mayaviMeshObject.mlab_source.set(scalars = scalarStress[:, 0])
                    lut = self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
                    lut[:, 0:3] = self.colormapHeat
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.data_range = (0, 1)

                    #self.surfaceObj.mayaviMeshObject.mlab_source.set(colormap='gist_heat')

            def _randomize_plate_colours_fired(self):
                '''
                When the button is pushed a new random colormap is created and the objects which might use it is updated.
                :return:
                '''
                self.colormapRandom = np.random.randint(0, 255, (256, 3))

                if self.surface_scalar_mode == 'ID':
                    lut = self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
                    lut[:, 0:3] = self.colormapRandom
                    self.surfaceObj.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut

                lut = self.borderObj.module_manager.scalar_lut_manager.lut.table.to_array()
                lut[:, 0:3] = self.colormapRandom
                self.borderObj.module_manager.scalar_lut_manager.lut.table = lut
                self.mayaviScene.mlab.draw()

            def _simulate_button_fired(self):

                '''
                i = 0
                #t = time.time()
                #max_framerate = 10
                while 1:
                    i += 1
                    print(i)
                    self._flow_toggle_fired()
                    #while time.time() - t < (1. / max_framerate):
                    #    pass
                    #t = time.time()

                    time.sleep(5)
                '''


                print('animation starts')
                @mlab.animate(delay=500, ui = True)
                def anim():
                    for iStep in range(10):
                        print(iStep)
                        # s.mlab_source.scalars = np.asarray(x * 0.1 * (i + 1), 'd')
                        iPlate = -1
                        for key, plate in plateDictionary.items():
                            iPlate += 1
                            if iPlate < 100:
                                # print(plate.averageFlowVector)
                                # print(Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0])))
                                # plate.averageFlowVector /= Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0]))
                                # for iPoint in range(plate.numberOfVertices):
                                #    plate.vertices[iPoint, :] = Utility.RotateAroundAxis(plate.vertices[iPoint, :],
                                #                                                         plate.averageFlowVector, 1 * np.pi / 180)
                                #    #plate.vertices[iPoint, :] = Utility.RotateAroundAxis(plate.vertices[iPoint, :],
                                #    #                                                     plate.verticesFlow[iPoint, :], 1 * np.pi / 180)
                                # print(Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0])))
                                # avgVecNorm = Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0]))
                                # avgVec = plate.averageFlowVector / avgVecNorm
                                # plate.vertices = Utility.RotateAroundAxis(plate.vertices, plate.averageFlowVector, 1 * np.pi / 180)
                                # plate.vertices = Utility.RotateAroundAxis(plate.vertices, avgVec, 3*avgVecNorm * np.pi / 180)
                                # for iPoint in range(plate.numberOfVertices):
                                #     plate.vertices[iPoint, :] /= Utility.VectorDistance(plate.vertices[iPoint, :], np.array([0, 0, 0]))

                                plate.Rotate(angleScaling=3)
                                # plate.NormalizeVertices() # Unsure if this i needed.
                                plate.UpdateFlow()
                                plate.UpdateAverage()
                                # break
                            else:
                                break
                        # quit()
                        # print('------------------------------------------------------')

                        nPoints = 0
                        iPlate = -1
                        borderIndex = 0
                        for key, plate in plateDictionary.items():
                            iPlate += 1
                            if iPlate < 100:
                                if plate.numberOfVertices > 4:
                                    # plateVerts.append(plate.vertices)
                                    nBorderPoints = np.size(plate.borderVertex, 0)
                                    if iPlate == 0:
                                        plateVerts = plate.vertices
                                        plateFaces = plate.triangles
                                        meshScalarsID = plate.ID * np.ones((plate.numberOfVertices, 1))
                                        meshScalarsNumber = iPlate * np.ones((plate.numberOfVertices, 1))
                                        scalarStress = plate.stressVector
                                        scalarInteractionStress = plate.interactionStress
                                        # meshScalarsNumber = plate.ID * np.ones((plate.numberOfVertices, 1))
                                        borderVertices = plate.borderVertex
                                        borderLines = plate.borderLines
                                        borderScalars = iPlate * np.ones((nBorderPoints, 1))
                                        secondBorderScalars = iPlate * np.ones(
                                            (np.size(plate.secondBorderIndices, 0), 1))
                                        secondBorderVertices = plate.secondBorderPoints
                                    else:
                                        plateVerts = np.append(plateVerts, plate.vertices, axis=0)
                                        plateFaces = np.append(plateFaces, plate.triangles + nPoints, axis=0)
                                        meshScalarsID = np.append(meshScalarsID,
                                                                  plate.ID * np.ones((plate.numberOfVertices, 1)),
                                                                  axis=0)
                                        meshScalarsNumber = np.append(meshScalarsNumber,
                                                                      iPlate * np.ones((plate.numberOfVertices, 1)),
                                                                      axis=0)
                                        scalarStress = np.append(scalarStress, plate.stressVector, axis=0)
                                        scalarInteractionStress = np.append(scalarInteractionStress,
                                                                            plate.interactionStress, axis=0)

                                        borderVertices = np.append(borderVertices, plate.borderVertex, axis=0)
                                        borderLines = np.append(borderLines, borderIndex + plate.borderLines, axis=0)
                                        borderScalars = np.append(borderScalars, iPlate * np.ones((nBorderPoints, 1)),
                                                                  axis=0)
                                        secondBorderScalars = np.append(secondBorderScalars,
                                                                        iPlate * np.ones(
                                                                            (np.size(plate.secondBorderIndices, 0), 1)),
                                                                        axis=0)
                                        secondBorderVertices = np.append(secondBorderVertices, plate.secondBorderPoints,
                                                                         axis=0)
                                    nPoints += plate.numberOfVertices
                                    borderIndex += nBorderPoints
                            else:
                                break

                        # visObj.mayaviMeshObject.mlab_source.scalars = s
                        self.surfaceObj.mayaviMeshObject.mlab_source.x = plateVerts[:, 0]
                        self.surfaceObj.mayaviMeshObject.mlab_source.y = plateVerts[:, 1]
                        self.surfaceObj.mayaviMeshObject.mlab_source.z = plateVerts[:, 2]

                        yield

                import mayavi
                a = anim()
                #self.mayaviScene.Focus()
                #a.close()
                #help(a)
                #t = mayavi.tools.Animator(500, a.next)
                #t.edit_traits()

                #a.Close()
                #help(a)
                #quit()
                #a.Start()

                #mlab.show()

                #quit()




            def flow_change(self):

                self.flowVisibility = 1.0 - self.flowVisibility
                #print(self.flowVisibility)
                self.flowObj.mayaviFlowObject.mlab_source.set(x = (self.flowVisibility + 0.02) * world.vertices[:, 0],
                                                              y = (self.flowVisibility + 0.02) * world.vertices[:, 1],
                                                              z = (self.flowVisibility + 0.02) * world.vertices[:, 2])

                #pts.mlab_source.reset(x=_x, y=_y, z=_z, )
                #yield
                #self.mayaviScene.draw()

            def _flow_toggle_fired(self):
                '''
                Moves the flowfield inside/outside of the sphere, thus making it visible/unvisible to the user.
                :return:
                '''

                import wx

                def animate_sleep(x):
                    n_steps = int(x / 0.01)
                    for i in range(n_steps):
                        time.sleep(0.01)
                        wx.Yield()

                for i in range(10):
                    animate_sleep(1)

                    self.flowVisibility = 1.0 - self.flowVisibility
                    print(self.flowVisibility)
                    self.flowObj.mayaviFlowObject.mlab_source.set(
                        x=(self.flowVisibility + 0.02) * world.vertices[:, 0],
                        y=(self.flowVisibility + 0.02) * world.vertices[:, 1],
                        z=(self.flowVisibility + 0.02) * world.vertices[:, 2])


                '''
                @mlab.show
                @mlab.animate()
                def anim_flow():
                    for i in range(10):
                        self.flowVisibility = 1.0 - self.flowVisibility
                        print(self.flowVisibility)
                        self.flowObj.mayaviFlowObject.mlab_source.set(
                            x=(self.flowVisibility + 0.02) * world.vertices[:, 0],
                            y=(self.flowVisibility + 0.02) * world.vertices[:, 1],
                            z=(self.flowVisibility + 0.02) * world.vertices[:, 2])
                        yield


                anim_flow()
                '''
                #self.flowVisibility = 1.0 - self.flowVisibility
                #print(self.flowVisibility)
                #self.flowObj.mayaviFlowObject.mlab_source.set(x = (self.flowVisibility + 0.02) * world.vertices[:, 0],
                #                                              y = (self.flowVisibility + 0.02) * world.vertices[:, 1],
                #                                              z = (self.flowVisibility + 0.02) * world.vertices[:, 2])
                '''
                    #self.mayaviScene.reset_zoom()
                    #time.sleep(5)
                if self.a < 10:
                    #print(a)
                    self.a += 1
                    self._flow_toggle_fired()
                    time.sleep(5)
                else:
                    self.a = 0
                    print('-------')
                    return
                '''


            def _quit_button_fired(self):
                quit()

            view = View(
                Item('mayaviScene', height = 0.9, show_label = False, editor = SceneEditor(), tooltip = 'Test tooltip message'),
                HGroup('simulate_button', 'flow_toggle', 'surface_scalar_mode', 'randomize_plate_colours', 'quit_button'),
                resizable = True)
        print('>------------------------------<')
        print('>---< GUI has been created >---<')
        print('>------------------------------<')
        mayaviWindow = MayaviWindow(colormapRandom, colormapHeat)
        mayaviWindow.configure_traits()
                   #MayaviWindow(colormapRandom, colormapHeat).configure_traits()
        mlab.show()
        quit()








        '''
        from numpy import ogrid, sin

        from traits.api import HasTraits, Instance
        from traitsui.api import View, Item

        from mayavi.sources.api import ArraySource
        from mayavi.modules.api import IsoSurface

        from mayavi.core.ui.api import SceneEditor, MlabSceneModel


        class MayaviView(HasTraits):

            scene = Instance(MlabSceneModel, ())

            # The layout of the panel created by Traits
            view = View(Item('scene', editor=SceneEditor(), resizable=True,
                             show_label=False),
                        resizable=True)

            def __init__(self):
                HasTraits.__init__(self)
                # Create some data, and plot it using the embedded scene's engine
                x, y, z = ogrid[-10:10:100j, -10:10:100j, -10:10:100j]
                scalars = sin(x * y * z) / (x * y * z)
                src = ArraySource(scalar_data=scalars)
                self.scene.engine.add_source(src)
                src.add_module(IsoSurface())


        # -----------------------------------------------------------------------------
        # Wx Code
        import wx


        class MainWindow(wx.Frame):

            def __init__(self, parent, id):
                wx.Frame.__init__(self, parent, id, 'Mayavi in Wx')
                self.mayavi_view = MayaviView()
                # Use traits to create a panel, and use it as the content of this
                # wx frame.
                self.control = self.mayavi_view.edit_traits(
                    parent=self,
                    kind='subpanel').control
                self.Show(True)


        app = wx.PySimpleApp()
        frame = MainWindow(None, wx.ID_ANY)
        app.MainLoop()
        quit()
        '''

    # Mayavi window with a picker.
    if False:
        nPoints = 0
        iPlate = -1
        for key, plate in plateDictionary.items():
            iPlate += 1
            if iPlate < 100:
                if plate.numberOfVertices > 4:
                    #plateVerts.append(plate.vertices)
                    if iPlate == 0:
                        plateVerts = plate.vertices
                        plateFaces = plate.triangles
                        meshScalarsID = plate.ID * np.ones((plate.numberOfVertices, 1))
                        meshScalarsNumber = iPlate * np.ones((plate.numberOfVertices, 1))
                        #meshScalarsNumber = plate.ID * np.ones((plate.numberOfVertices, 1))
                    else:
                        plateVerts = np.append(plateVerts, plate.vertices, axis = 0)
                        plateFaces = np.append(plateFaces, plate.triangles+nPoints, axis = 0)
                        meshScalarsID = np.append(meshScalarsID, plate.ID * np.ones((plate.numberOfVertices, 1)), axis = 0)
                        meshScalarsNumber = np.append(meshScalarsNumber, iPlate * np.ones((plate.numberOfVertices, 1)), axis=0)
                        #meshScalarsNumber = np.append(meshScalarsNumber, plate.ID * np.ones((plate.numberOfVertices, 1)),axis=0)
                    nPoints += plate.numberOfVertices
            else:
                break
        mayaviWindow = Visualization.MayaviWindow(windowSize=System_Info.SCREEN_RESOLUTION,
                                                  squaredWindow=True)
        #mlab.points3d(plateVerts[:, 0], plateVerts[:, 1], plateVerts[:, 2], scale_factor = 0.01)
        '''
        visObjBlack = Visualization.VisualizeGlobe(vertices=0.98*world4.vertices.copy(),
                                              faces=world4.faces.copy(),
                                              radius=world4.radius.copy(),
                                              scalars=world4.radius.copy(),
                                              projectTopography=True,
                                              projectRadiusSpan=[1, 1.03],
                                              interpolatedTriangleColor=True,
                                              colormap='gist_earth',
                                              figure = mayaviWindow.figure)
        '''
        customColorMap = np.random.randint(0, 255, (256, 3))

        #customColorMap[0, 0:3] = 0

        meshScalarsNumber /= np.max(meshScalarsNumber)
        meshScalarsNumber *= 255
        meshScalarsNumber = np.round(meshScalarsNumber)
        #print(meshScalarsNumber)

        pointColor = np.zeros((np.size(plateVerts, 0), 3))
        for iLoop, loopScalar in enumerate(meshScalarsNumber):
            pointColor[iLoop, :] = customColorMap[int(loopScalar), :]
        pointColor /= 255


        '''
        platePointVis = mlab.points3d(plateVerts[:, 0], plateVerts[:, 1], plateVerts[:, 2], scale_factor = 0.01)
        platePointVis.glyph.scale_mode = 'scale_by_vector'
        platePointVis.mlab_source.dataset.point_data.scalars = meshScalarsNumber[:, 0]/255
        lut = platePointVis.module_manager.scalar_lut_manager.lut.table.to_array()
        lut[:, 0:3] = customColorMap
        platePointVis.module_manager.scalar_lut_manager.lut.table = lut
        '''

        visObj = Visualization.VisualizeGlobe(vertices=plateVerts,
                                              faces=plateFaces,
                                              radius=np.ones((nPoints, 1)),
                                              scalars=meshScalarsNumber,
                                              projectTopography=True,
                                              projectRadiusSpan=[1, 1.03],
                                              interpolatedTriangleColor=True,
                                              colormap='gist_earth',
                                              customColormap=customColorMap,
                                              figure = mayaviWindow.figure)

        #print('-----VISUALISATION DONE-----')
        #mlab.show()
        #quit()

        cursor3d = mlab.points3d(0., 0., 0.,
                                 color=(1, 1, 1),
                                 scale_factor=0.025,)
        cursorLine = mlab.plot3d([0, 0.1], [0, 0.1], [0, 0.1])
        textObj = mlab.text(0.8, 0.85, '-', width = 0.2)
        # The textobject should be rotated in respect to the picked point and the camera, thusly locking it in place from
        # the perspective of the camera. The textObject could also be created as a separate panel.
        def picker_callback(picker_obj):
            picked = picker_obj.actors
            '''

            #print(picker_obj)
            #print('============================================================')
            #print(picked)
            #quit()

            #print(picker_obj)
            print('--------------')

            print(picker_obj.pick_position)
            queryResult = world.kdTree.query(picker_obj.pick_position)
            #print(queryResult)
            vertexID = queryResult[1]
            pickedPlateID = meshScalarsID[vertexID][0]
            #print(pickedPlateID)
            pickedPlate = plateDictionary[pickedPlateID]
            averageStress = np.mean(pickedPlate.stressVector)
            maximumStress = np.max(pickedPlate.stressVector)
            tmpStr = 'Plate ID          : ' + str(int(pickedPlateID)) + \
                     '\n' + '# points          : ' + str(pickedPlate.numberOfVertices) + \
                     '\n' + 'mean(Stress) : ' + str(round(averageStress, 3)) + \
                     '\n' + 'Max(stress)    : ' + str(round(maximumStress, 3))
            #cursor3d.mlab_source.reset(x=plateVerts[vertexID, 0],
            #                           y=plateVerts[vertexID, 1],
            #                           z=plateVerts[vertexID, 2])
            r = picker_obj.pick_position
            cursorLine.mlab_source.trait_set(x = [0, 1.5*r[0]],
                                             y = [0, 1.5*r[1]],
                                             z = [0, 1.5*r[2]])
            #cursorLine.mlab_source.trait_set(x = [0, 1.5*r[0]])
            #cursor3d.mlab_source.reset(x=r[0],
            #                           y=r[1],
            #                           z=r[2])
            textObj.text = tmpStr
            '''
            #if mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
            if visObj.mayaviMeshObject.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
                # m.mlab_source.points is the points array underlying the vtk
                # dataset. GetPointId return the index in this array.

                #print(picker_obj.point_id)
                #print('---------------------')
                #print('Mesh ID : ', picker_obj.point_id)

                print(picker_obj.point_id)

                cursor3d.mlab_source.reset(x=plateVerts[picker_obj.point_id, 0],
                                           y=plateVerts[picker_obj.point_id, 1],
                                           z=plateVerts[picker_obj.point_id, 2])
                #pickedPlateID = meshScalarsID[picker_obj.point_id][0]
                pickedPlateID = meshScalarsID[picker_obj.point_id][0]
                pickedPlate = plateDictionary[pickedPlateID]
                averageStress = np.mean(pickedPlate.stressVector)
                maximumStress = np.max(pickedPlate.stressVector)

                tmpStr =        'Plate ID          : ' + str(int(pickedPlateID)) +\
                         '\n' + '# points          : ' + str(pickedPlate.numberOfVertices) +\
                         '\n' + 'mean(Stress) : ' + str(round(averageStress, 3)) +\
                         '\n' + 'Max(stress)    : ' + str(round(maximumStress, 3))
                textObj.text = tmpStr
            

        pickerObj = mayaviWindow.figure.on_mouse_pick(picker_callback)
        pickerObj.tolerance = 0.01

        print('-----VISUALISATION DONE-----')
        mlab.show()
        quit()





    nPoints = 0
    iPlate = -1
    for key, plate in plateDictionary.items():
        iPlate += 1
        if iPlate < 2:
            if plate.numberOfVertices > 4:
                # plateVerts.append(plate.vertices)
                if iPlate == 0:
                    plateVerts = plate.vertices
                    plateFaces = plate.triangles
                    meshScalarsNumber = plate.ID * np.ones((plate.numberOfVertices, 1))
                else:
                    plateVerts = np.append(plateVerts, plate.vertices, axis=0)
                    plateFaces = np.append(plateFaces, plate.triangles + nPoints, axis=0)
                    meshScalarsNumber = np.append(meshScalarsNumber, plate.ID * np.ones((plate.numberOfVertices, 1)), axis=0)
                nPoints += plate.numberOfVertices
        else:
            break

    customColorMap = np.random.randint(0, 255, (256, 3))
    mayaviWindow = Visualization.MayaviWindow(windowSize = System_Info.SCREEN_RESOLUTION,
                                              squaredWindow = True)
    #figure = mayaviWindow.figure
    visObj = Visualization.VisualizeGlobe(
                                          vertices = plateVerts,
                                          faces = plateFaces,
                                          radius = np.ones((nPoints, 1)),
                                          scalars = meshScalarsNumber,
                                          projectTopography = True,
                                          projectRadiusSpan = [1, 1.03],
                                          interpolatedTriangleColor = True,
                                          colormap = 'gist_earth',
                                          customColormap = customColorMap)
    #Visualization.VisualizeGlobe(vertices=world.vertices.copy(),
    #                             faces=world.faces.copy(),
    #                             radius=world.radius.copy(),
    #                             scalars=s,
    #                             projectTopography=True,
    #                             projectRadiusSpan=[1, 1.03],
    #                             interpolatedTriangleColor=True,
    #                             colormap='gist_earth',
    #                             randomColormap=True)
    #visObj = Visualization.VisualizeGlobe(vertices=world.vertices.copy(),
    #                             faces=world.faces.copy(),
    #                             radius=world.radius.copy(),
    #                             scalars=s,
    #                             projectTopography=True,
    #                             projectRadiusSpan=[1, 1.03],
    #                             interpolatedTriangleColor=True,
    #                             colormap='gist_earth',
    #                             randomColormap=True,
    #                             windowSize = System_Info.SCREEN_RESOLUTION,
    #                             squaredWindow = True)
    #Visualization.VisualizeFlow(world.vertices,
    #                            plateCollection.xFlow,
    #                            plateCollection.yFlow,
    #                            plateCollection.zFlow,
    #                            world.faces,
    #                            newFigure=False,
    #                            sizeFactor=0.02)
    Visualization.VisualizeFlow(1.02*world4.vertices,
                                plateCollection.xFlow[treeResult[1]],
                                plateCollection.yFlow[treeResult[1]],
                                plateCollection.zFlow[treeResult[1]],
                                world4.faces,
                                newFigure=False,
                                sizeFactor=0.08)


    #print('visialization done')
    #mlab.show()
    #quit()

    padding = len(str(100))

    mlab.view(azimuth=0, elevation=90, distance=4, focalpoint='auto',
              roll=0, reset_roll=True, figure=mayaviWindow.figure)

    @mlab.animate(delay = 10)
    def anim():
        for iStep in range(100):
            #s.mlab_source.scalars = np.asarray(x * 0.1 * (i + 1), 'd')
            iPlate = -1
            for key, plate in plateDictionary.items():
                iPlate += 1
                if iPlate < 2:
                    #print(plate.averageFlowVector)
                    #print(Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0])))
                    #plate.averageFlowVector /= Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0]))
                    #for iPoint in range(plate.numberOfVertices):
                    #    plate.vertices[iPoint, :] = Utility.RotateAroundAxis(plate.vertices[iPoint, :],
                    #                                                         plate.averageFlowVector, 1 * np.pi / 180)
                    #    #plate.vertices[iPoint, :] = Utility.RotateAroundAxis(plate.vertices[iPoint, :],
                    #    #                                                     plate.verticesFlow[iPoint, :], 1 * np.pi / 180)
                    #print(Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0])))
                    # avgVecNorm = Utility.VectorDistance(plate.averageFlowVector, np.array([0, 0, 0]))
                    # avgVec = plate.averageFlowVector / avgVecNorm
                    #plate.vertices = Utility.RotateAroundAxis(plate.vertices, plate.averageFlowVector, 1 * np.pi / 180)
                    # plate.vertices = Utility.RotateAroundAxis(plate.vertices, avgVec, 3*avgVecNorm * np.pi / 180)
                    # for iPoint in range(plate.numberOfVertices):
                    #     plate.vertices[iPoint, :] /= Utility.VectorDistance(plate.vertices[iPoint, :], np.array([0, 0, 0]))

                    plate.Rotate(angleScaling = 3)
                    #plate.NormalizeVertices() # Unsure if this i needed.
                    plate.UpdateFlow()
                    plate.UpdateAverage()
                    #break
                else:
                    break
            #quit()
            #print('------------------------------------------------------')

            '''
            iPlate = -1
            for key, plate in plateDictionary.items():
                iPlate += 1
                if iPlate < 2:
                    if iPlate == 0:
                        v = plate.vertices
                        s = plate.ID * np.ones((plate.numberOfVertices, 1))
                    else:
                        v = np.append(v, plate.vertices, axis=0)
                        s = np.append(s, plate.ID * np.ones((plate.numberOfVertices, 1)), axis=0)
                else:
                    break

                # plate.ID
            v = np.append(v, 0.98 * world.vertices, axis=0)
            s = np.append(s, np.zeros((world.numberOfvertices, 1)), axis=0)

            i = scipy.interpolate.NearestNDInterpolator(v, s)
            s = i(world.vertices)
            '''
            s = np.zeros((world.numberOfvertices, 1))
            iPlate = -1
            for key, plate in plateDictionary.items():
                iPlate += 1
                if iPlate < 2:
                    for iPoint in range(plate.numberOfVertices):
                        s[plate.nearestPointIndex[iPoint]] = plate.ID
                        s[plate.secondNearestPointIndex[iPoint]] = plate.ID
                    if iPlate == 0:
                        plateVerts = plate.vertices
                    else:
                        plateVerts = np.append(plateVerts, plate.vertices, axis = 0)
                else:
                    break


            #visObj.mayaviMeshObject.mlab_source.scalars = s
            visObj.mayaviMeshObject.mlab_source.x = plateVerts[:, 0]
            visObj.mayaviMeshObject.mlab_source.y = plateVerts[:, 1]
            visObj.mayaviMeshObject.mlab_source.z = plateVerts[:, 2]


            # Saves a screenshot as a .png file.
            if saveScreenshots:
                zeros = '0' * (padding - len(str(iStep)))
                fileName = Root_Directory.Path() + '/Movies/screenshots/anim' + zeros + str(iStep) + '.tiff'
                mlab.savefig(filename=fileName)

            yield



            #mlab.start_recording() # Look into recording of animations.
            # mlab.move() # Could possibly be used to track the plate.
            #help(mlab.start_recording)
            #quit()

    '''
    import cProfile
    pr = cProfile.Profile()
    pr.enable()

    # Call anim() here
    anim()
    pr.disable()
    pr.print_stats(2)
    quit()
    '''

    anim()
    mlab.show()
    quit()

    print('---- VISUALIZATION DONE ----')
    mlab.show()
    quit()






# Generates the perlin flow, the function returns 3 distinct vectors.
xFlow, yFlow, zFlow = Simulation.Noise.PerlinNoise3DFlow(4,
                                                         world.vertices.copy(),
                                                         1,
                                                         amplitudeScaling = 1.5,
                                                         projectOnSphere = True,
                                                         normalizedVectors = True)
if False:
    # Creates unitvectors (phi and theta).
    thetaUnitVectors = np.zeros((world.numberOfvertices, 3))
    phiUnitVectors = np.zeros((world.numberOfvertices, 3))
    thetaVector = np.zeros((world.numberOfvertices, 1))
    phiVector = np.zeros((world.numberOfvertices, 1))
    for iPoint, point in enumerate(world.vertices):
        phiVector[iPoint], thetaVector[iPoint], radius = Utility.CaartesianToSpherical(point)
        # Phi unit vector
        phiUnitVectors[iPoint, 0] = -np.sin(phiVector[iPoint])
        phiUnitVectors[iPoint, 1] = np.cos(phiVector[iPoint])
        phiUnitVectors[iPoint, 2] = 0
        # Theta unit vector
        thetaUnitVectors[iPoint, 0] = -np.cos(phiVector[iPoint])*np.sin(thetaVector[iPoint])
        thetaUnitVectors[iPoint, 1] = -np.sin(phiVector[iPoint])*np.sin(thetaVector[iPoint])
        thetaUnitVectors[iPoint, 2] = np.cos(thetaVector[iPoint])


if False:
    flowVectors = np.append(np.reshape(xFlow, (world.numberOfvertices, 1)), np.reshape(yFlow, (world.numberOfvertices, 1)), axis=1)
    flowVectors = np.append(flowVectors, np.reshape(zFlow, (world.numberOfvertices, 1)), axis=1)
    flowVectorsRotated = flowVectors.copy()
    fixedPoint = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]  # [1/np.sqrt(2), 1/(np.sqrt(2)), 0]

    print(np.sqrt(fixedPoint[0] ** 2 + fixedPoint[1] ** 2 + fixedPoint[2] ** 2))
    for iVertex in range(world.numberOfvertices):
        tmp = Utility.RotateVector2Steps(world.vertices[iVertex, :].copy(), fixedPoint, flowVectors[iVertex, :].copy())
        flowVectorsRotated[iVertex, :] = tmp

tic = time.clock()

xFlow = np.reshape(xFlow, (world.numberOfvertices, 1))
yFlow = np.reshape(yFlow, (world.numberOfvertices, 1))
zFlow = np.reshape(zFlow, (world.numberOfvertices, 1))
flowVectors = np.append(xFlow, yFlow, axis=1)
flowVectors = np.append(flowVectors, zFlow, axis=1)

if True:
    # These are the rotations exises for each vertex.
    flowRotationVectors = np.zeros((world.numberOfvertices, 3))
    for iVertex, flow in enumerate(flowVectors):
        flowRotationVectors[iVertex, :] = np.cross(world.vertices[iVertex, :], flow)
def f():
    identityMatrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Used when rotating flow vectors.
    #numberOfSets = 20 # Not used.
    numberOfPlatesEachSet = 1000
    plateID = -1*np.ones((world.numberOfvertices, 1))
    rLimit = 0.2  #0.4
    stepLimit = 10500#10#50

    # |---- rLimit-angle table ----|
    # rLimit     angle (degrees)
    # 0.1        5.73
    # 0.2        11.48
    # 0.3        17.25
    # 0.4        23.07
    # 0.5        28.96
    # 0.6        34.92
    # 0.7        40.97
    # 0.8        47.16
    # 0.9        53.49
    # 1.0        60.00
    # 1.1        66.73
    # 1.2        73.74
    # 1.3        81.08
    # sqrt(2)    90
    # This angle represents the maximum deviation allowed for points to be considered to be of the same plate
    #

    plateIndexLocal = np.zeros((world.numberOfvertices, 1))
    plateListTmp = []#[[] for i in range(numberOfPlatesTotal)]


    availablePoints = [i for i in np.linspace(0, world.numberOfvertices-1, world.numberOfvertices, dtype=int)]

    iRun = -1
    #for iRun in range(numberOfSets):
    while np.size(availablePoints) > 0:
        iRun += 1
        plateListSet = [[] for i in range(numberOfPlatesEachSet)]
        #plateMeanPoints = np.zeros((numberOfPlatesEachSet, 3))
        #plateMeanFlows = np.zeros((numberOfPlatesEachSet, 3))
        for iPlateSet in range(numberOfPlatesEachSet):
            initialPoint = [np.random.randint(0, world.numberOfvertices)]

            if np.size(availablePoints) > 0:
                initialPoint = np.random.choice(availablePoints)
                availablePoints.remove(initialPoint)
            else:
                # No more plates should be created.
                break

            plateListSet[int(iPlateSet)] = [initialPoint]
            #plateMeanPoints[int(iPlateSet), :] = world.vertices[initialPoint, :]
            #plateMeanFlows[int(iPlateSet), :] = flowVectors[initialPoint, :]
            #print(initialPoint)
            plateID[initialPoint] = iRun*numberOfPlatesEachSet + iPlateSet
        #print(plateListSet)
        print('==========================================================')
        #quit()
        for iStep in range(stepLimit):
            for iPlate, iPlateSet, plate in zip(np.linspace(iRun * numberOfPlatesEachSet, (iRun + 1) * numberOfPlatesEachSet-1,
                    numberOfPlatesEachSet), range(numberOfPlatesEachSet), plateListSet):

                    if int(plateIndexLocal[int(iPlate)]) < np.size(plate):
                    #if plateIndexLocal[int(iPlate)] < np.size(plate):
                        adjacentPoints = world.neighbours.IDList[plate[int(plateIndexLocal[int(iPlate)])]][0]
                        for p in adjacentPoints:

                            if True:
                                # x-/y-/zFlow should be rotated before calculating r
                                if True:
                                    if False:
                                        # Flow vectors. The candidate flow vector will be rotated to the position of the main flow vector.
                                        mainFlow = flowVectors[plate[0], :].copy()
                                        #mainFlow = plateMeanFlows[iPlateSet, :]
                                        candidateFlow = flowVectors[p, :].copy()

                                        # The main point, and the candidate point which might be added.
                                        mainPoint = world.vertices[plate[0], :].copy()
                                        #mainPoint = plateMeanPoints[iPlateSet, :]
                                        candidatePoint = world.vertices[p, :].copy()

                                        #candidateFlow = Utility.RotateVector2Steps(candidatePoint, mainPoint, candidateFlow)
                                        candidateFlow = Utility.RotateVector(candidatePoint, mainPoint, candidateFlow)

                                        #print('flow vector norm: ', np.sqrt(mainFlow[0]**2 + mainFlow[1]**2 + mainFlow[2]**2))
                                        # Calculates the norm between the vectors. Will be used to determine if the candidate should be
                                        # part of the existing plate.
                                        r = Utility.VectorDistance(mainFlow, candidateFlow)
                                    else:
                                        r = Utility.VectorDistance(flowRotationVectors[plate[0], :], flowRotationVectors[p, :])
                                else:
                                    if False:
                                        r = Utility.VectorDistance(flowVectorsRotated[plate[0], :], flowVectorsRotated[p, :])
                                    else:
                                        mainFlow = flowVectors[plate[0], :].copy()
                                        candidateFlow = flowVectors[p, :].copy()

                                        thetaMain = thetaUnitVectors[plate[0], :].copy()
                                        phiMain = phiUnitVectors[plate[0], :].copy()

                                        thetaCandidate = thetaUnitVectors[p, :].copy()
                                        phiCandidate = phiUnitVectors[p, :].copy()

                                        a = np.dot(mainFlow, thetaMain)
                                        b = np.dot(mainFlow, phiMain)

                                        c = np.dot(candidateFlow, thetaCandidate)
                                        d = np.dot(candidateFlow, phiCandidate)

                                        r = Utility.VectorDistance(np.array([a, b, 0]), np.array([c, d, 0]))

                                #r = np.sqrt((mainFlow[0] - candidateFlow[0]) ** 2 +
                                #            (mainFlow[1] - candidateFlow[1]) ** 2 +
                                #            (mainFlow[2] - candidateFlow[2]) ** 2)
                            else:
                                r = np.sqrt((xFlow[plate[0]] - xFlow[p]) ** 2 +
                                            (yFlow[plate[0]] - yFlow[p]) ** 2 +
                                            (zFlow[plate[0]] - zFlow[p]) ** 2)

                            #print(r)
                            #quit()
                            if r < rLimit and plateID[p] < 0:
                                #print(plateID[p])


                                #print(np.size(plateListSet[int(iPlateSet)]))

                                #plateMeanPointsOld = plateMeanPoints[iPlateSet, :].copy()
                                #plateMeanPoints[iPlateSet, :] = (plateMeanPoints[iPlateSet, :]*np.size(plateListSet[int(iPlateSet)])+
                                #                                 candidatePoint) / (np.size(plateListSet[int(iPlateSet)]) + 1)
                                #print(plateMeanPointsOld)
                                #print(plateMeanPoints[iPlateSet, :])
                                #print('----')
                                #tmp1 = Utility.RotateVector(plateMeanPointsOld, plateMeanPoints[iPlateSet, :], plateMeanFlows[iPlateSet, :])
                                #tmp2 = Utility.RotateVector(plateMeanPointsOld, plateMeanPoints[iPlateSet, :],candidateFlow)
                                #plateMeanFlows[iPlateSet, :] = (tmp1*np.size(plateListSet[int(iPlateSet)]) + tmp2) / (np.size(plateListSet[int(iPlateSet)]) + 1)
                                #plateMeanFlows[iPlateSet, :] /= np.sqrt(plateMeanFlows[iPlateSet, 0]**2 + plateMeanFlows[iPlateSet, 1]**2 + plateMeanFlows[iPlateSet, 2]**2)

                                #plateMeanFlows[iPlateSet, :] *= 0
                                #print(plate)
                                #for iTmpPlate, tmpPoint in enumerate(plate):
                                #    plateMeanFlows[iPlateSet, :] += (plateMeanFlows[iPlateSet, :] * iTmpPlate + flowVectors[tmpPoint, :]) / (iTmpPlate + 1)
                                #    plateMeanFlows[iPlateSet, :] /= np.sqrt(
                                #        plateMeanFlows[iPlateSet, 0] ** 2 + plateMeanFlows[iPlateSet, 1] ** 2 +
                                #        plateMeanFlows[iPlateSet, 2] ** 2)

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

        for plate in plateListSet:
            plateListTmp.append(plate)
        #break
        print('Number of available points : ', np.size(availablePoints))
    return plateListTmp

#from line_profiler import LineProfiler
#lp = LineProfiler()
#lp_wrapper = lp(f)
#lp_wrapper()
#lp.print_stats()
#quit()

plateListTmp = f()



#print(plateList)
#print('Number of available points : ', np.size(availablePoints))
#print(np.min(plateID))
#print(np.max(plateID))



# The plates should be compared and combined here.

minimumPlateSizePercentage = 0.002
minimumPlateSize = np.round(world.numberOfvertices*minimumPlateSizePercentage)
print('minimumPlateSize', minimumPlateSize)

plateDictionary = {}
Simulation.PlateDynamics.Plate.Initialize(minimumPlateSize = minimumPlateSize)

a = 0

for plate in plateListTmp:
    if np.size(plate)>0:
        #plateObject = Simulation.PlateDynamics.Plate(vertices = world.vertices[plate, :],
        #                                             thickness = rLimit)
        #plateObject = Simulation.PlateDynamics.Plate(vertices = world.vertices[plate, :],
        #                                             thickness = 1.2)
        plateObject = Simulation.PlateDynamics.Plate(vertices = world.vertices[plate, :],
                                                     thickness = 1.2)# 0.6
        #plateCollection.AddPlate(plateObject)
        plateDictionary[plateObject.ID] = plateObject
        a += np.size(plate)
'''
#for vertex in world.vertices:
for iVertex in range(world.numberOfvertices):
    vertex = world.vertices[iVertex:iVertex+1, :]
    plate = Simulation.PlateDynamics.Plate(vertices = vertex,
                                           thickness = rLimit)
    plateDictionary[plate.ID] = plate
'''
print('The initial plates has been created.')

#plateCollection = Simulation.PlateDynamics.PlateCollection(plateList)  # Used for storing a list of plates to file.
# All plates gets access to the list of all plates.
#Simulation.PlateDynamics.Plate.LinkLists(plateDictionary,
#                                         world.kdTree,
#                                         flowVectors,
#                                         world)
Simulation.PlateDynamics.Plate.LinkLists(plateDictionary,
                                         world.kdTree,
                                         flowRotationVectors,
                                         world)
# Creates/updates a KD-tree which is used to decide which plates are adjacent to which.
Simulation.PlateDynamics.Plate.UpdateKDTree()


for iPlate, plate in plateDictionary.items():
    plate.CalculateAdjacentPlates()
    plate.UpdateFlow() # Interpolates the flow onto the plate points.
    plate.UpdateAverage()
    plate.CalculateStress()
    #print(plate.adjacentPlateIDs)
    #print(plate.centerPoint)
    #print(plate.averageFlowVector)
    #print('------')

    #print(plateMeanFlows[iPlate, :])
    #print(plate.averageFlowVector)
    #print(plateMeanPoints[iPlate, :])
    #print(plate.centerPoint)
    #plateMeanPoints[iPlate, :] = plate.centerPoint
    #plateMeanFlows[iPlate, :] = plate.averageFlowVector
    #print('----------------')




iPlate = -1
for key, plate in plateDictionary.items():
    iPlate += 1
    if iPlate == 0:
        v = plate.vertices
        s = plate.ID*np.ones((plate.numberOfVertices, 1))
    else:
        v = np.append(v, plate.vertices, axis = 0)
        s = np.append(s, plate.ID*np.ones((plate.numberOfVertices, 1)), axis = 0)

    #plate.ID

i = scipy.interpolate.NearestNDInterpolator(v, s)
s = i(world.vertices)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = True)

Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
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

'''
# Used to visualize the flow within one plate, and the plates mean flow.
Visualization.VisualizeFlow(world.vertices,
                            xFlow[:, 0],
                            yFlow[:, 0],
                            zFlow[:, 0],
                            world.faces,
                            newFigure = True,
                            sizeFactor = 0.03)
Visualization.VisualizeFlow(plateDictionary[0].vertices,
                            plateDictionary[0].verticesFlow[:, 0],
                            plateDictionary[0].verticesFlow[:, 1],
                            plateDictionary[0].verticesFlow[:, 2],
                            newFigure = False,
                            sizeFactor = 0.04,
                            arrowColor = (0, 0, 1))
Visualization.VisualizeFlow(np.reshape(plateDictionary[0].centerPoint, (1, 3)),
                            plateDictionary[0].averageFlowVector[0],
                            plateDictionary[0].averageFlowVector[1],
                            plateDictionary[0].averageFlowVector[2],
                            newFigure = False,
                            sizeFactor = 0.06,
                            arrowColor = (1, 1, 1))
'''

toc = time.clock()
print('time in sec : ', toc-tic)
#print('Visualization done')
#mlab.show()
#quit()


print(len(Simulation.PlateDynamics.Plate.plateDictionary))
Simulation.PlateDynamics.Plate.CheckForMerge()
print(len(Simulation.PlateDynamics.Plate.plateDictionary))
iPlate = -1
for key, plate in plateDictionary.items():
    iPlate += 1
    if iPlate == 0:
        v = plate.vertices
        s = plate.ID*np.ones((plate.numberOfVertices, 1))
    else:
        v = np.append(v, plate.vertices, axis = 0)
        s = np.append(s, plate.ID*np.ones((plate.numberOfVertices, 1)), axis = 0)

    #plate.ID

i = scipy.interpolate.NearestNDInterpolator(v, s)
s = i(world.vertices)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = True)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
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
#print('============================')
#print('||>- Visualization done -<||')
#print('============================')
#mlab.show()
#quit()

ticPlateMerge = time.clock()
print(len(Simulation.PlateDynamics.Plate.plateDictionary))
for i in range(20):
    Simulation.PlateDynamics.Plate.CheckForMerge()
    print(len(Simulation.PlateDynamics.Plate.plateDictionary))

print('======================================')
for key, plate in Simulation.PlateDynamics.Plate.plateDictionary.items():
    plate.UpdateFlow() # Interpolates the flow onto the plate points.
    plate.UpdateAverage()
    plate.CalculateStress()
    print('# points       = ', plate.numberOfVertices)
    print('maximum stress = ', np.max(plate.stressVector))
    print('-----------------------')
tocPlateMarge = time.clock()

print('Time to merge plates: ', tocPlateMarge-ticPlateMerge)
#
# Write a method for breaking plates up.
#




#for iPlate, plate in plateDictionary.items():
#    print(plate.adjacentPlateIDs)

#quit()


#
# This code interpolates plate ID's onto the vertices of a icosphere. The sphere can then be visualized where each color
# correponds to a specific plate. This code should work even if the vertices of the plates are not aligned with the
# vertices of the icosphere.
#
iPlate = -1
for key, plate in plateDictionary.items():
    iPlate += 1
    if iPlate == 0:
        v = plate.vertices
        s = plate.ID*np.ones((plate.numberOfVertices, 1))
    else:
        v = np.append(v, plate.vertices, axis = 0)
        s = np.append(s, plate.ID*np.ones((plate.numberOfVertices, 1)), axis = 0)

    #plate.ID

i = scipy.interpolate.NearestNDInterpolator(v, s)
s = i(world.vertices)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
                             projectTopography = True,
                             projectRadiusSpan = [1, 1.03],
                             interpolatedTriangleColor = True,
                             colormap = 'gist_earth',
                             randomColormap = True)
Visualization.VisualizeGlobe(vertices = world.vertices.copy(),
                             faces = world.faces.copy(),
                             radius = world.radius.copy(),
                             scalars = s,
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



pltCol = Simulation.PlateDynamics.PlateCollection(plateDictionary = plateDictionary,
                                                  xFlow = xFlow[:, 0],
                                                  yFlow = yFlow[:, 0],
                                                  zFlow = zFlow[:, 0])

import Root_Directory
fileName = Root_Directory.Path() + '/Templates/Plate_Collection_' + str(1.2) + '.pkl'
fileToOpen = open(fileName, 'wb')
pickle.dump(pltCol, fileToOpen, pickle.HIGHEST_PROTOCOL)
fileToOpen.close()


print('============================')
print('||>- Visualization done -<||')
print('============================')
mlab.show()
quit()

# When comparing two plates a new plate is created. Right now the flow vectors are queried from a kd-tree, instead they
# could be given as input to the constructor, in a simular manner as is the case with teh vertices.


print('a = ', a)


# The plates should be compared and combined here.
toc = time.clock()
print('time in sec : ', toc-tic)

#print(plateList)
#print(plateList[0])
#world.vertices[]


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







