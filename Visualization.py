from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group

from tvtk.api import tvtk
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor

import numpy as np



class MayaviWindow(HasTraits):
    '''
    The class consists of a window with two scenes. One scene visualizes the initial terrain and the other scene
    visualizes the changed/updated (eroded) terrain. Buttons shall be added in order to toogle on/off things
    like: water, sediment, rock layers, rainfall density etc.
    '''

    # In the original scene the terrain is visualized as it looked before the erosion simulation was done. The updated
    # scene visualizes the terrain as it looks after the erosion simulation.
    originalScene = Instance(MlabSceneModel, ())
    updatedScene = Instance(MlabSceneModel, ())


    # The layout of the window. (two horizontaly paralell scenes)
    view = View(HSplit(
            Group(
                Item('updatedScene',
                     editor=SceneEditor(), height=300, width=400),
                show_labels=False
                 ),
            Group(
                Item('originalScene',
                     editor=SceneEditor(), height=300, width=400),
                show_labels=False
                 ),
                      ),
                        resizable=True,
                        title='Terrain Visualizer',
                )


    def Surf(self, z, scene = 'updated', type = 'terrain'):
        # Depending on input selects which scene to plot in.
        if scene == 'updated':
            figureToPlotIn = self.updatedScene.mayavi_scene
        else:
            figureToPlotIn = self.originalScene.mayavi_scene

        # Depending on input specifies settings like colour and opacity for the surface.
        if type == 'terrain':
            surfaceColour = (0.55, 0.45, 0.35)
            surfaceOpacity = 1
        else:
            if type == 'water':
                surfaceColour = (0.1, 0.3, 0.5)
                surfaceOpacity = 0.5

        # Adds the surface to the selected scene.
        mlab.surf(z, figure=figureToPlotIn, color=surfaceColour, opacity=surfaceOpacity)


    # The @on_trait_change() methods are called when configure_traits() is called.
    # A text annotation is created for each scene here. Since the figure input of the mlab.text() method do not work
    # properly a dummy surface is used to get around this problem.
    @on_trait_change('originalScene.activated')
    def OriginalSceneSettings(self):
        self.originalScene.interactor.interactor_style = \
            tvtk.InteractorStyleTerrain()
        # The distance value is set in such a way that the entire surface should be visible without need to zoom out.
        self.originalScene.scene.mlab.view(30, 60, distance=2.5*512)
        mlab.surf(np.zeros([2, 2]), figure = self.originalScene.mayavi_scene, opacity = 0) # Dummy surface
        mlab.text(0.7, 0.9, 'Initial', width=0.3)

    @on_trait_change('updatedScene.activated')
    def UpdatedSceneSettings(self):
        self.updatedScene.interactor.interactor_style = \
            tvtk.InteractorStyleTerrain()
        # The distance value is set in such a way that the entire surface should be visible without need to zoom out.
        self.updatedScene.scene.mlab.view(30, 60, distance=2.5*512)
        mlab.surf(np.zeros([2, 2]), figure = self.updatedScene.mayavi_scene, opacity = 0) # Dummy surface
        mlab.text(0.6, 0.9, 'Eroded', width=0.4)




