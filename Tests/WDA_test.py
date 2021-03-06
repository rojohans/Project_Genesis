'''
From this sqript file the waterdrop algorithm can be tested.

The program will create a severe bug if parallell drops are used (numberOfDrops >1).
    # In the case of parallell drops the adjacentHeights and the heightDifference needs to be recalculated, these
    # values depends on the heightMap which could have been changed by another drop. If this is not taken into
    # account multiple drops can erode the same area creating a deep hole. (the depth of the hole will approach -inf).
    # An alternative approach is to limit the amount of deposition/erosion to 1/N of the highest allowed
    # deposition/erosion value for each drop. N is the number of drops, in that case it would require all N drops
    # to deposit/erode all the material, thusly no holes or peaks could be formed. A downside is that it lowers the
    # effect of each drop, thusly requiring more drops to get the same effect. If the code utilizes a GPU for
    # parallell computations this approach could be viable.


    The inertia part of the DiscreteDrop class needs to be altered. Perhaps the previous direction should be included when choosing new direction, using the inertia as weights.

    When choosing "topdown" in the visualizer the plot object generated should not be a 3D surface.
'''

import Simulation.Erosion.Hydrolic.WDA as WDA
import Simulation.Noise as Noise
import Visualization
import numpy as np
import time # Used to time code, like tic-toc.
import cProfile


mapSize = 512
initialMaximumHeight = 100
numberOfRuns = 1
numberOfDrops = 1 # This do not need to be 1 but changing it does not result in true parallel drops.
numberOfSteps = 64
maximumErosionRadius = 10  # This determines how many erosion templates should be created.


#100000  = 20 minuter
#1000000 = 3 timmar
#2000000 = 6 timmar


displaySurface = True
displayTrail = False
performProfiling = False


# The height map is generated from "simple noise".
heightMap = Noise.SimpleNoise(mapSize,3,1.8)
heightMap *= initialMaximumHeight
'''
# THIS CODE GENERATES INITIAL HILLS/MOUNTAINS
heightMap = np.zeros((mapSize, mapSize))
for x in range(mapSize):
    for y in range(mapSize):
        heightMap[x, y] += 1/(1 + 0.0002*((x-mapSize/2)**2 + (y-mapSize/2)**2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 3) ** 2 + (y - mapSize / 1.5) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 1.2) ** 2 + (y - mapSize / 3.1) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 1.2) ** 2 + (y - mapSize / 1.8) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 2.3) ** 2 + (y - mapSize / 2.8) ** 2))
        heightMap[x, y] += 1 / (1 + 0.0002 * ((x - mapSize / 2.9) ** 2 + (y - mapSize / 3.6) ** 2))
heightMap -= np.min(heightMap)
heightMap[heightMap < 0.2] = 0.2
a = Noise.SimpleNoise(mapSize,3,1.1)
a *= 0.2
heightMap += a
heightMap -= np.min(heightMap)
heightMap /= np.max(heightMap)
heightMap *= initialMaximumHeight
'''

initialRockMap = heightMap.copy() # Used to determine where sediment has accumulated.
initialSedimentMap = np.zeros((mapSize, mapSize))
initialTotalMap = heightMap.copy()

rockMap = initialRockMap.copy()
sedimentMap = initialSedimentMap.copy()
totalHeightMap = heightMap.copy() + sedimentMap.copy()
print('Noise has been generated')


# Creates a mayavi window and visualizes the initial terrain.
window = Visualization.MayaviWindow()
window.Surf(initialRockMap, type='terrain', scene='original')


# Creates templates used by all the drops.
WDA.WaterDrop.LinkToHeightMap(heightMap, sedimentMap, totalHeightMap)
WDA.WaterDrop.InitializeErosionTemplates(maximumErosionRadius)
WDA.ContinuousDrop.InitializeAdjacentTileTemplate()
WDA.DiscreteDrop.InitializeAdjacentTileTemplate()

'''
a = WDA.ContinuousDrop()

print(a)

import pickle

with open('test_save.pkl', 'wb') as output:
    pickle.dump(a, output, pickle.HIGHEST_PROTOCOL)

del a

with open('test_save.pkl', 'rb') as input:
    b = pickle.load(input)

print(b)

quit()
'''


if performProfiling:
    pr = cProfile.Profile()
    pr.enable()


print('Amount of material before simulation: %s' % np.sum(heightMap))
tic = time.time()
for iRun in range(numberOfRuns):
    # Create the drops
    drops = [WDA.ContinuousDrop(
                           numberOfSteps=numberOfSteps,
                           storeTrail=displayTrail,
                           inertia=0.3,
                           capacityMultiplier=200,
                           depositionRate=0.1,
                           erosionRate=0.01,
                           erosionRadius=4,
                           maximumUnimprovedSteps = 5) for index in range(numberOfDrops)]
    WDA.WaterDrop.LinkToDrops(drops)
    # Performs the drop simulation, step by step.
    for iStep in range(numberOfSteps):
        for drop in drops:
            drop()


toc = time.time()
print('elapsed time : %s sec' % (toc - tic))
print('Amount of material after simulation: %s' % np.sum(heightMap))


print(np.min(heightMap))
print(np.max(heightMap))


print(np.min(sedimentMap))
print(np.max(sedimentMap))

print(np.min(totalHeightMap))
print(np.max(totalHeightMap))

'''
from PIL import Image
import random


#a = np.zeros((16, 16, 3))
a = np.zeros((512, 512, 3))

a[:, :, 0] = 255*heightMap/np.max(heightMap)

img = Image.fromarray(a.astype('uint8'), 'RGB')

img.show()
'''


if performProfiling:
    pr.disable()
    pr.print_stats(2)



import Storage
import pickle
import datetime
now=datetime.datetime.now()

print(str(now))
#print(now.strftime("%Y-%m-%d %H:%M"))

#'Worlds/' + str(now) + '.pkl'


world = Storage.World(initialRockMap, initialSedimentMap, initialTotalMap, heightMap, sedimentMap, totalHeightMap)
pickle.dump(world, open('Worlds/' + now.strftime("%Y-%m-%d %H:%M") + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


# Visualizes the eroded terrain.
if displaySurface:
    #window.Surf(totalHeightMap, type='terrain', scene='original')
    window.Surf(heightMap, type='terrain', scene='updated')
    #sedimentMap = heightMap-unchangedHeightMap
    #sedimentMap[sedimentMap > 0] = 0
    #heightMapToPlot = unchangedHeightMap + sedimentMap
    #window.Surf(heightMapToPlot, type='terrain', scene='updated')

    #sedimentMap = heightMap-unchangedHeightMap
    #sedimentMap[sedimentMap < 0] = 0
    #c = unchangedHeightMap + sedimentMap
    #c[sedimentMap == 0] = 0


    #window.Surf(30+np.zeros([mapSize, mapSize]), type='water', scene='updated')
    #window.Surf(c, type='water', scene='updated')

    np.min(sedimentMap)
    heightMap[sedimentMap == 0] = 0
    window.Surf(heightMap + sedimentMap, type='sediment', scene='updated')
    #window.SedimentColour(sedimentMap)

    window.configure_traits()
    #mapSurface.Update(heightMap)


# The trails of the drops are visualized.
if displayTrail:
    pass
    #trailData = Visualization.PrepareLines(drops)
    #trailLines.Update(trailData)


if displaySurface:
    pass
    #print('Animation is done')
    #mainWindow.Keep(True)







'''
if False:
    import numpy as np

    from traits.api import HasTraits, Instance, Button, \
        on_trait_change
    from traitsui.api import View, Item, HSplit, Group

    from tvtk.api import tvtk
    from mayavi import mlab
    from mayavi.core.ui.api import MlabSceneModel, SceneEditor


    class MyDialog(HasTraits):

        scene1 = Instance(MlabSceneModel, ())
        scene2 = Instance(MlabSceneModel, ())

        # The layout of the window.
        view = View(HSplit(
                      Group(
                           Item('scene1',
                                editor=SceneEditor(), height=300,
                                width=400),
                           show_labels=False,
                      ),
                      Group(
                           Item('scene2',
                                editor=SceneEditor(), height=300,
                                width=400),
                           show_labels=False
                      ),
                    ),
                    resizable=True,
                    )


        def redraw_scene(self, scene, z):
            # Clears the selected scene and updates the visual objects.
            #mlab.clf(figure=scene.mayavi_scene)
            mlab.surf(z, figure=scene.mayavi_scene, color = (0.6, 0.4, 0.3))
            mlab.surf(30 + np.zeros([512, 512]),
                      opacity=0.5,
                      color=(0.1, 0.3, 0.7))


        # The @on_trait_change() methods are called when configure_traits() is called.
        @on_trait_change('scene1.activated')
        def display_scene1(self):
            print('scene 1 settings')

            self.scene1.interactor.interactor_style = \
                tvtk.InteractorStyleTerrain()
            self.scene1.scene.mlab.view(40, 130)
            self.redraw_scene(self.scene1, self.heightMap)

        @on_trait_change('scene2.activated')
        def display_scene2(self):
            print('scene 2 settings')

            self.scene2.interactor.interactor_style = \
                tvtk.InteractorStyleTerrain()
            self.scene2.scene.mlab.view(40, 130)
            self.redraw_scene(self.scene2, self.heightMap)


    m = MyDialog()
    m.heightMap = heightMap
    #m.redraw_scene(m.scene1, heightMap)
    #m.redraw_scene(m.scene2, heightMap)
    m.configure_traits() # This works similar to matplotlib.pyplot.show()
'''











'''
import numpy as np

from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, on_trait_change
from traitsui.api import View, Item, HSplit, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel


################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array

    # The position of the view
    position = Array(shape=(3,))

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    #scene_x = Instance(MlabSceneModel, ())
    #scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # The cursors on each view:
    cursors = Dict()

    disable_render = Bool

    _axis_names = dict(x=0, y=1, z=2)

    #---------------------------------------------------------------------------
    # Object interface
    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        #self.ipw_3d_x
        #self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _position_default(self):
        return 0.5*np.array(self.data.shape)

    def _data_src_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene,
                            name='Data',)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name,
                        name='Cut %s' % axis_name)
        #ipw = mlab.surf(self.data)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbacks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        print('scene settings')
        #outline = mlab.pipeline.outline(self.data_src,
        #                figure=self.scene3d.mayavi_scene,
        #                )
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HSplit(
                  Group(
                       Item('scene_z',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene3d',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title='Volume Slicer',
                )

################################################################################

# Create some data
data = heightMap

m = VolumeSlicer(data = data)
print('Object has been created')
m.configure_traits()
'''









'''
import numpy as np

from traits.api import HasTraits, Instance, Array, \
    Bool, Dict, on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel


################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    data = Array

    # The position of the view
    position = Array(shape=(3,))

    # The 4 views displayed
    scene3d = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    # The cursors on each view:
    cursors = Dict()

    disable_render = Bool

    _axis_names = dict(x=0, y=1, z=2)

    #---------------------------------------------------------------------------
    # Object interface
    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _position_default(self):
        return 0.5*np.array(self.data.shape)

    def _data_src_default(self):
        return mlab.pipeline.scalar_field(self.data,
                            figure=self.scene3d.mayavi_scene,
                            name='Data',)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src,
                        figure=self.scene3d.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name,
                        name='Cut %s' % axis_name)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbacks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d.activated')
    def display_scene3d(self):
        outline = mlab.pipeline.outline(self.data_src,
                        figure=self.scene3d.mayavi_scene,
                        )
        self.scene3d.mlab.view(40, 50)
        # Interaction properties can only be changed after the scene
        # has been created, and thus the interactor exists
        for ipw in (self.ipw_3d_x, self.ipw_3d_y, self.ipw_3d_z):
            ipw.ipw.interaction = 0
        self.scene3d.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        self.update_position()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)
        scene.scene.parallel_projection = True
        ipw_3d   = getattr(self, 'ipw_3d_%s' % axis_name)

        # We create the image_plane_widgets in the side view using a
        # VTK dataset pointing to the data on the corresponding
        # image_plane_widget in the 3D view (it is returned by
        # ipw_3d._get_reslice_output())
        side_src = ipw_3d.ipw._get_reslice_output()
        ipw = mlab.pipeline.image_plane_widget(
                            side_src,
                            plane_orientation='z_axes',
                            vmin=self.data.min(),
                            vmax=self.data.max(),
                            figure=scene.mayavi_scene,
                            name='Cut view %s' % axis_name,
                            )
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Extract the spacing of the side_src to convert coordinates
        # into indices
        spacing = side_src.spacing

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0

        x, y, z = self.position
        cursor = mlab.points3d(x, y, z,
                            mode='axes',
                            color=(0, 0, 0),
                            scale_factor=2*max(self.data.shape),
                            figure=scene.mayavi_scene,
                            name='Cursor view %s' % axis_name,
                        )
        self.cursors[axis_name] = cursor

        # Add a callback on the image plane widget interaction to
        # move the others
        this_axis_number = self._axis_names[axis_name]
        def move_view(obj, evt):
            # Disable rendering on all scene
            position = list(obj.GetCurrentCursorPosition()*spacing)[:2]
            position.insert(this_axis_number, self.position[this_axis_number])
            # We need to special case y, as the view has been rotated.
            if axis_name is 'y':
                position = position[::-1]
            self.position = position

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                                        self._axis_names[axis_name]]

        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)

        # Some text:
        mlab.text(0.01, 0.8, axis_name, width=0.08)

        # Choose a view that makes sens
        views = dict(x=(0, 0), y=(90, 180), z=(0, 0))
        mlab.view(views[axis_name][0],
                  views[axis_name][1],
                  focalpoint=0.5*np.array(self.data.shape),
                  figure=scene.mayavi_scene)
        scene.scene.camera.parallel_scale = 0.52*np.mean(self.data.shape)

    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # Traits callback
    #---------------------------------------------------------------------------
    @on_trait_change('position')
    def update_position(self):
        """ Update the position of the cursors on each side view, as well
            as the image_plane_widgets in the 3D view.
        """
        # First disable rendering in all scenes to avoid unecessary
        # renderings
        self.disable_render = True

        # For each axis, move image_plane_widget and the cursor in the
        # side view
        for axis_name, axis_number in self._axis_names.items():
            ipw3d = getattr(self, 'ipw_3d_%s' % axis_name)
            ipw3d.ipw.slice_position = self.position[axis_number]

            # Go from the 3D position, to the 2D coordinates in the
            # side view
            position2d = list(self.position)
            position2d.pop(axis_number)
            if axis_name is 'y':
                position2d = position2d[::-1]
            # Move the cursor
            # For the following to work, you need Mayavi 3.4.0, if you
            # have a less recent version, use 'x=[position2d[0]]'
            self.cursors[axis_name].mlab_source.trait_set(
                x=position2d[0], y=position2d[1], z=0)

        # Finally re-enable rendering
        self.disable_render = False

    @on_trait_change('disable_render')
    def _render_enable(self):
        for scene in (self.scene3d, self.scene_x, self.scene_y,
                                                  self.scene_z):
            scene.scene.disable_render = self.disable_render


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                       Item('scene_y',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene_z',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene_x',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       Item('scene3d',
                            editor=SceneEditor(scene_class=Scene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title='Volume Slicer',
                )


################################################################################

# Create some data
x, y, z = np.ogrid[-5:5:100j, -5:5:100j, -5:5:100j]
data = np.sin(3*x)/x + 0.05*z**2 + np.cos(3*y)

m = VolumeSlicer(data=data)
m.configure_traits()
'''

















'''
from mayavi import mlab
from tvtk.api import tvtk

#help(mlab.figure)

fig = mlab.figure(size=(700, 700))
fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

mlab.surf(30+np.zeros([mapSize, mapSize]),
          opacity = 0.5,
          color = (0.1, 0.3, 0.7))
mlab.surf(heightMap,
          color = (0.6, 0.4, 0.3))



secondFigure = mlab.figure()
secondFigure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

mlab.surf(heightMap,
          color = (0.6, 0.4, 0.3))
mlab.show()
'''









# This code could be very useful when buttons and sliders are to be added to the project.
'''
import numpy as np

from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor


class MyDialog(HasTraits):

    scene1 = Instance(MlabSceneModel, ())
    scene2 = Instance(MlabSceneModel, ())

    button1 = Button('Redraw')
    button2 = Button('Redraw')

    @on_trait_change('button1')
    def redraw_scene1(self):
        self.redraw_scene(self.scene1)

    @on_trait_change('button2')
    def redraw_scene2(self):
        self.redraw_scene(self.scene2)

    def redraw_scene(self, scene):
        # Notice how each mlab call points explicitely to the figure it
        # applies to.
        mlab.clf(figure=scene.mayavi_scene)
        x, y, z, s = np.random.random((4, 100))
        mlab.points3d(x, y, z, s, figure=scene.mayavi_scene)

    # The layout of the dialog created
    view = View(HSplit(
                  Group(
                       Item('scene1',
                            editor=SceneEditor(), height=250,
                            width=300),
                       'button1',
                       show_labels=False,
                  ),
                  Group(
                       Item('scene2',
                            editor=SceneEditor(), height=250,
                            width=300),
                       'button2',
                       show_labels=False
                  ),
                ),
                resizable=True,
                )


m = MyDialog()
m.configure_traits() # This works similar to matplotlib.pyplot.show()
'''
















print('The program has ended')
