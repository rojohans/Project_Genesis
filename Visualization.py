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
            surfaceColour = (0.5, 0.45, 0.4)
            surfaceOpacity = 1
        else:
            if type == 'water':
                surfaceColour = (0.1, 0.3, 0.5)
                surfaceOpacity = 0.5
            else:
                if type == 'sediment':
                    surfaceColour = (0.25, 0.15, 0.15)
                    surfaceOpacity = 1

        # Adds the surface to the selected scene.
        #if type == 'sediment':
        #    #z[z==np.min(z)]+=50
        #    #z += 50
        #   self.a = mlab.surf(z, figure=figureToPlotIn, opacity=surfaceOpacity)
        #else:
        mlab.surf(z, figure=figureToPlotIn, color=surfaceColour, opacity=surfaceOpacity)


    def SedimentColour(self, z):
        # The method changes the texture of the sediment surface. The texture is such that color of each element is
        # directly correlated to the sediment depth of the element.
        # ============== BUG ============== BUG ============== BUG ============== BUG ============== BUG ==============
        # There is a bug which occurs randomly, in those cases the texture is transposed. It is unclear what causes the
        # bug and what can be done to remove it.
        # ============== BUG ============== BUG ============== BUG ============== BUG ============== BUG ==============
        from PIL import Image
        b = np.zeros((512, 512, 3))
        b[:, :, 1] = 255 * z / np.max(z)
        img = Image.fromarray(b.astype('uint8'), 'RGB')
        #img.show()
        img = img.rotate(90) # The rotation is necessary for the image to align with the surface properly.
        #img = img.transpose(Image.TRANSPOSE)  # The rotation is necessary for the image to align with the surface properly.
        img.save('my.png')

        bmp1 = tvtk.PNGReader()
        #bmp1 = tvtk.JPEGReader()
        bmp1.file_name = "my.png"
        my_texture = tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

        # If the scalar_visibility is not False the colour of the texture will depend on the height of the surface. When
        # they value is false the appearance of the texture do not depend on the height of the surface, THIS IS CRUCIAL.
        self.a.actor.mapper.scalar_visibility = False
        self.a.actor.enable_texture = True
        self.a.actor.tcoord_generator_mode = 'plane'
        self.a.actor.actor.texture = my_texture
        #self.a.actor.texture_source_object = img
        #self.a.actor.texture.interpolate = True
        #self.a.actor.texture.repeat = False


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

