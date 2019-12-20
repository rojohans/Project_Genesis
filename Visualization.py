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
                    surfaceColour = (0.5, 0.45, 0.4)
                    #(0.25, 0.15, 0.15)
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

class VisualizeGlobe():
    def __init__(self,
                 vertices,
                 faces,
                 radius,
                 scalars,
                 projectTopography = True,
                 projectRadiusSpan = [1, 1.1],
                 interpolatedTriangleColor = False,
                 colormap = 'gist_earth',
                 newFigure = True,
                 randomColormap = False,
                 windowSize = [700, 700],
                 squaredWindow = False):
        # ------------------------------------------------------------------------------------------
        # Creates a triangular mesh object visualizing a globe given the inputs. The surface can be projected onto a sphere
        # or drawn as an irregular sphere (different radius for different vertices).
        #
        # projectTopography: If True will give a sphere with a scalar map drawn on it.
        # interpolatedTriangleColor: If True will give triangles with varying color thoughout the triangle. The color will
        #                            be interpolated between the edge scalar values. If False will give triangles with a
        #                            single color. The color will be the mean of the edge scalar values.
        # squaredWindow : If true will make the mayavi window
        # ------------------------------------------------------------------------------------------

        if squaredWindow:
            windowSize = [np.min(windowSize), np.min(windowSize)]

        if newFigure:
            self.figure = mlab.figure(size = windowSize)
            self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        if interpolatedTriangleColor:
            if projectTopography:
                self.mayaviMeshObject = mlab.triangular_mesh(vertices[:, 0] * 0.99,
                                            vertices[:, 1] * 0.99,
                                            vertices[:, 2] * 0.99,
                                            faces,
                                            scalars = scalars[:, 0],
                                            colormap=colormap)
            else:
                radius *= (projectRadiusSpan[1]-projectRadiusSpan[0])
                self.mayaviMeshObject = mlab.triangular_mesh(vertices[:, 0] * (projectRadiusSpan[0] + radius),
                                            vertices[:, 1] * (projectRadiusSpan[0] + radius),
                                            vertices[:, 2] * (projectRadiusSpan[0] + radius),
                                            faces,
                                            scalars=scalars[:, 0],
                                            colormap=colormap)
        else:
            if projectTopography:
                self.mayaviMeshObject = mlab.triangular_mesh(vertices[:, 0] * 0.99,
                                            vertices[:, 1] * 0.99,
                                            vertices[:, 2] * 0.99,
                                            faces,
                                            representation='wireframe',
                                            opacity=0,
                                            colormap=colormap)
            else:
                radius *= (projectRadiusSpan[1]-projectRadiusSpan[0])
                self.mayaviMeshObject = mlab.triangular_mesh(vertices[:, 0] * (projectRadiusSpan[0] + radius),
                                            vertices[:, 1] * (projectRadiusSpan[0] + radius),
                                            vertices[:, 2] * (projectRadiusSpan[0] + radius),
                                            faces,
                                            representation='wireframe',
                                            opacity=0,
                                            colormap=colormap)
            faceHeight = np.mean(scalars[faces], 1)
            self.mayaviMeshObject.mlab_source.dataset.cell_data.scalars = faceHeight
            self.mayaviMeshObject.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
            self.mayaviMeshObject.mlab_source.update()
            self.mayaviMeshObject = mlab.pipeline.set_active_attribute(self.mayaviMeshObject, cell_scalars='Cell data')
            self.mayaviMeshObject = mlab.pipeline.surface(self.mayaviMeshObject, colormap=colormap)

            '''
            #mesh.mlab_source.dataset.point_data.scalars = radius[:, 0]
            #mesh.mlab_source.dataset.point_data.scalars.name = 'Point data'
            #mesh.mlab_source.update()
            #mesh2 = mlab.pipeline.set_active_attribute(mesh, point_scalars='Point data')
            #mlab.pipeline.surface(mesh2, colormap=colormap)

            # Check the tvtk.polydata documentation. Do this to change the resolution of the triangles (polygons/cells).
            #print(mesh.mlab_source.dataset)
            #help(mesh.mlab_source.dataset)
            '''

            #help(mlab.pipeline.surface)
            #mlab.pipeline.surface(mesh2, color = (0.5, 0.8, 0.1))
            #help(mlab.pipeline.set_active_attribute)
            #help(mesh.mlab_source.dataset.point_data.interpolate_allocate)



        if randomColormap is True:
            lut = self.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table.to_array()
            lut[:, 0:3] =  np.random.randint(0, 255, (256, 3))
            lut[0, 0:3] = 0
            self.mayaviMeshObject.module_manager.scalar_lut_manager.lut.table = lut


class VisualizeFlow():
    def __init__(self,
                 vertices,
                 xFlow,
                 yFlow,
                 zFlow,
                 backgroundFaces = None,
                 arrowColor = (1, 0, 0),
                 sizeFactor = 0.02,
                 newFigure = True):
        # ------------------------------------------------------------------------------------------
        # Creates a quiver object visualizing a 3d flow. A black inner sphere can be used in order to make the flow
        # vectors more distinct.
        #
        # backgroundFaces: An optional input. A list of triangles which span an entire sphere, the faces must correspond
        #                  to the vertices. If this input is given a black sphere will be drawn within the flow
        #                  vectors, this improves visibility.
        # ------------------------------------------------------------------------------------------
        if newFigure:
            self.figure = mlab.figure()
            self.figure.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        # Creates a black sphere "within" the flow arrows, this make the arrows more visible.
        if backgroundFaces is not None:
            self.mayaviMeshObject = mlab.triangular_mesh(vertices[:, 0] * 0.95,
                                                         vertices[:, 1] * 0.95,
                                                         vertices[:, 2] * 0.95,
                                                         backgroundFaces,
                                                         color = (0, 0, 0))

        self.mayaviFlowObject = mlab.quiver3d(vertices[:, 0],
                                              vertices[:, 1],
                                              vertices[:, 2],
                                              xFlow,
                                              yFlow,
                                              zFlow,
                                              resolution=3,
                                              scale_factor=sizeFactor,
                                              color=arrowColor,
                                              mode='arrow')#arrow #cone










