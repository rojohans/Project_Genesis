
'''
====:====:====:====:====:====:====:====:====:====:====:====:====:====
====:====:====:====:====:====:====:====:====:====:====:====:====:====
====:====:====:====:====:====:====:====:====:====:====:====:====:====
'''

import numpy as np
from mayavi.mlab import *
from traits.api import HasTraits, Instance, Button, \
    on_trait_change
from traitsui.api import View, Item, HSplit, Group
from tvtk.api import tvtk
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor

from scipy.spatial import SphericalVoronoi
from scipy.spatial import Delaunay


# <<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>
# Consider settings the radius to 1 and using a 3D delanauy triangulation in order to get the triangles to visualize. In
# this way a coordinate transformation is not needed.
# <<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>--<<--<<==>>-->>


numberOfHotSpots = 30
numberOfThetaValues = 50 # Used to generate the plate particles

u = np.transpose(np.random.rand(numberOfHotSpots, 1))
v = np.transpose(np.random.rand(numberOfHotSpots, 1))

randomPhiAngles = 2 * np.pi * u
randomThetaAngles = np.arccos((2 * v - 1)) - np.pi/2

# [x, y, z]
hotSpotCoordinates = np.zeros((numberOfHotSpots, 3))
hotSpotCoordinates[:, 0] = np.cos(randomPhiAngles) * np.cos(randomThetaAngles)
hotSpotCoordinates[:, 1] = np.sin(randomPhiAngles) * np.cos(randomThetaAngles)
hotSpotCoordinates[:, 2] = np.sin(randomThetaAngles)


thetaVector = np.linspace(-np.pi/2, np.pi/2, numberOfThetaValues)
dTheta = thetaVector[1] - thetaVector[0]
numberOfParticles = int(np.sum(np.fmax(np.round(np.cos(thetaVector) * 2 * np.pi / dTheta), np.ones(np.shape(thetaVector)))))
# particleCoordinates = [x, y, z, r, phi, theta]
wrapParticles = [None for i in range(numberOfThetaValues)]
particleCoordinates = np.zeros((numberOfParticles, 6))
hotSpotID = np.linspace(0, numberOfHotSpots-1, numberOfHotSpots)
voronoiID = np.zeros((numberOfParticles, 1))
iParticle = 0
iWrapParticle = 0
for theta in thetaVector:
    #phiVector = np.linspace(0, 2 * np.pi, np.sum(np.ceil(2 * np.pi * np.cos(theta) / dTheta)))
    phiVector = np.linspace(0, 2 * np.pi * (1 - 1/np.max((np.round(np.cos(theta)*2*np.pi/dTheta), 1))),
                            np.max((np.round(np.cos(theta) * 2 * np.pi / dTheta), 1)))
    #phiVector = np.linspace(0, 2 * np.pi * (1 - 1 / np.max((np.ceil(np.cos(theta) * 2 * np.pi / dTheta), 1))),
    #                        np.max((np.ceil(np.cos(theta) * 2 * np.pi / dTheta), 1)))
    #phiVector += (phiVector[1]-phiVector[0])/2 # Shifts the phi values such that there
    #phiVector += (phiVector[1, 0]-phiVector[0, 0])/2 # Shifts the phi values such that there
    #phiVector = 2*np.pi*np.random.rand(int(np.sum(np.ceil(2 * np.pi * np.cos(theta) / dTheta))), 1)
    for phi in phiVector:
        r = 1 + 0.2 * np.random.rand()
        #phi *= 1 + 0.1*np.random.rand()
        particleCoordinates[iParticle, 0] = r * np.cos(phi) * np.cos(theta)
        particleCoordinates[iParticle, 1] = r * np.sin(phi) * np.cos(theta)
        particleCoordinates[iParticle, 2] = r * np.sin(theta)
        particleCoordinates[iParticle, 3] = r
        particleCoordinates[iParticle, 4] = phi
        particleCoordinates[iParticle, 5] = theta


        r = np.sqrt((particleCoordinates[iParticle, 0] - hotSpotCoordinates[:, 0]) ** 2 +
                    (particleCoordinates[iParticle, 1] - hotSpotCoordinates[:, 1]) ** 2 +
                    (particleCoordinates[iParticle, 2] - hotSpotCoordinates[:, 2]) ** 2)
        voronoiID[iParticle, 0] = hotSpotID[r == np.min(r)]
        iParticle += 1
    wrapParticles[iWrapParticle] = iParticle-1
    iWrapParticle += 1

center = np.array([0, 0, 0])
print('Particle coordinates has been generated')

#(cos(theta)+1)/ = v
# Performs a delanauay triangulation in order create the surface.
# Performs a scaling of the angles in order to avoid artifacts at the poles.
#np.arccos((2 * v - 1)) - np.pi/2

'''
Consider wrapping all the points. This would enable new triangles which would bind together the formation. After the new
triangles has been created any duplicate triangles should be discarded (this might not be neccessary).
'''

u = np.random.rand(np.size(particleCoordinates[:, 3]), 1)
v = np.random.rand(np.size(particleCoordinates[:, 3]), 1)
p = u*2*np.pi
t = np.arccos((2 * v - 1)) - np.pi/2
#print(np.shape(p))
#print(np.shape(t))
#print(np.shape(particleCoordinates[:, 3]))

#particleCoordinates[:, 4] = particleCoordinates[:, 4]/(2*np.pi)
#np.sign(particleCoordinates[:, 5])*
#particleCoordinates[:, 5] = (np.cos(particleCoordinates[:, 5]+np.pi/2)+1)/2


#print(wrapParticles)
#print(particleCoordinates[wrapParticles, 4])

#print(np.min(particleCoordinates[:, 4]))
#print(np.max(particleCoordinates[:, 4]))
#print(np.min(particleCoordinates[:, 5]))
#print(np.max(particleCoordinates[:, 5]))
#tri = Delaunay(particleCoordinates[:, 4:6])


particleCoordinates[:, 0] /= particleCoordinates[:, 3]
particleCoordinates[:, 1] /= particleCoordinates[:, 3]
particleCoordinates[:, 2] /= particleCoordinates[:, 3]
tri = Delaunay(particleCoordinates[:, 0:3])
#print(tri.simplices)
#print(np.shape(tri.simplices))
particleCoordinates[:, 0] *= particleCoordinates[:, 3]
particleCoordinates[:, 1] *= particleCoordinates[:, 3]
particleCoordinates[:, 2] *= particleCoordinates[:, 3]
#print(tri)
#print(tri.simplices)
sv = SphericalVoronoi(particleCoordinates[:, 0:3], 1, center)

print('Delaunay triangulation has been performed')


fig1 = mlab.figure()
# Makes the scene such that the z-axis is always pointing up.
fig1.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

#pts = mlab.points3d(particleCoordinates[:, 0]/particleCoordinates[:, 3], particleCoordinates[:, 1]/particleCoordinates[:, 3], particleCoordinates[:, 2]/particleCoordinates[:, 3], scale_factor=0.01)
#mesh1 = mlab.pipeline.delaunay3d(pts)
#surf = mlab.pipeline.surface(mesh1)

mlab.points3d(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], scale_factor = 0.001, color = (1, 1, 1))
#mlab.triangular_mesh(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], tri.simplices[:, 1:4], color = (0.5, 0.5, 0.5))
heightSurface = mlab.triangular_mesh(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], tri.simplices[:, 1:4], scalars=voronoiID[:, 0])
#mlab.points3d(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], scale_factor = 0.01, color=(1, 0, 0)) # Voronoi vertices

# Changes the colormap used by the height surface.
lut = heightSurface.module_manager.scalar_lut_manager.lut.table.to_array()
lut[:, 0] = 255*np.random.rand(256)
lut[:, 1] = 255*np.random.rand(256)
lut[:, 2] = 255*np.random.rand(256)
heightSurface.module_manager.scalar_lut_manager.lut.table = lut


fig2 = mlab.figure()
fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
mlab.points3d(particleCoordinates[:, 4], particleCoordinates[:, 5], particleCoordinates[:, 3]/particleCoordinates[:, 3], scale_factor = 0.01, color = (0, 0, 0))

#fig3 = mlab.figure()
#fig3.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(p[:, 0], t[:, 0], particleCoordinates[:, 3]/particleCoordinates[:, 3], scale_factor = 0.01, color = (0, 0, 0))
#mlab.points3d(np.cos(p)*np.cos(t), np.sin(p)*np.cos(t), np.sin(t), scale_factor = 0.01, color = (0, 0, 0))

#u = p/(2*np.pi)
#v = (np.cos(t+np.pi/2)+1)/2
#mlab.points3d(u[:, 0], v[:, 0], particleCoordinates[:, 3]/particleCoordinates[:, 3], scale_factor = 0.01, color = (0, 0, 0))

#fig2 = mlab.figure()
#fig2.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
#mlab.points3d(particleCoordinates[:, 4], particleCoordinates[:, 5], particleCoordinates[:, 3], scale_factor = 0.05, color = (0.7, 0.2, 0.1))
#mlab.triangular_mesh(particleCoordinates[:, 4], particleCoordinates[:, 5], particleCoordinates[:, 3], tri.simplices, color = (0.5, 0.5, 0.5))

#mlab.points3d(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], scale_factor = 0.1, color = (1, 0, 0))
mlab.show()
quit()


'''
# LOOKS INTERESTING BUT REQUIRES MATPLOTLIB. CHANGES NEED TO BE DONE TO THE ANACONDA ENVIRONMENT.
    # Author: S. Chris Colbert <sccolbert@gmail.com>
    # Copyright (c) 2009, S. Chris Colbert
    # License: BSD Style

    from __future__ import print_function

    # this import is here because we need to ensure that matplotlib uses the
    # wx backend and having regular code outside the main block is PyTaboo.
    # It needs to be imported first, so that matplotlib can impose the
    # version of Wx it requires.
    import matplotlib

    matplotlib.use('WXAgg')
    import pylab as pl

    import numpy as np
    from mayavi import mlab
    from mayavi.core.ui.mayavi_scene import MayaviScene


    def get_world_to_view_matrix(mlab_scene):
        """returns the 4x4 matrix that is a concatenation of the modelview transform and
        perspective transform. Takes as input an mlab scene object."""

        if not isinstance(mlab_scene, MayaviScene):
            raise TypeError('argument must be an instance of MayaviScene')

        # The VTK method needs the aspect ratio and near and far clipping planes
        # in order to return the proper transform. So we query the current scene
        # object to get the parameters we need.
        scene_size = tuple(mlab_scene.get_size())
        clip_range = mlab_scene.camera.clipping_range
        aspect_ratio = float(scene_size[0]) / float(scene_size[1])

        # this actually just gets a vtk matrix object, we can't really do anything with it yet
        vtk_comb_trans_mat = mlab_scene.camera.get_composite_projection_transform_matrix(
            aspect_ratio, clip_range[0], clip_range[1])

        # get the vtk mat as a numpy array
        np_comb_trans_mat = vtk_comb_trans_mat.to_array()

        return np_comb_trans_mat


    def get_view_to_display_matrix(mlab_scene):
        """ this function returns a 4x4 matrix that will convert normalized
            view coordinates to display coordinates. It's assumed that the view should
            take up the entire window and that the origin of the window is in the
            upper left corner"""

        if not (isinstance(mlab_scene, MayaviScene)):
            raise TypeError('argument must be an instance of MayaviScene')

        # this gets the client size of the window
        x, y = tuple(mlab_scene.get_size())

        # normalized view coordinates have the origin in the middle of the space
        # so we need to scale by width and height of the display window and shift
        # by half width and half height. The matrix accomplishes that.
        view_to_disp_mat = np.array([[x / 2.0, 0., 0., x / 2.0],
                                     [0., -y / 2.0, 0., y / 2.0],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])

        return view_to_disp_mat


    def apply_transform_to_points(points, trans_mat):
        """a function that applies a 4x4 transformation matrix to an of
            homogeneous points. The array of points should have shape Nx4"""

        if not trans_mat.shape == (4, 4):
            raise ValueError('transform matrix must be 4x4')

        if not points.shape[1] == 4:
            raise ValueError('point array must have shape Nx4')

        return np.dot(trans_mat, points.T).T


    if __name__ == '__main__':
        f = mlab.figure()

        N = 4

        # create a few points in 3-space
        X = np.random.random_integers(-3, 3, N)
        Y = np.random.random_integers(-3, 3, N)
        Z = np.random.random_integers(-3, 3, N)

        # plot the points with mlab
        pts = mlab.points3d(X, Y, Z)

        # now were going to create a single N x 4 array of our points
        # adding a fourth column of ones expresses the world points in
        # homogenous coordinates
        W = np.ones(X.shape)
        hmgns_world_coords = np.column_stack((X, Y, Z, W))

        # applying the first transform will give us 'unnormalized' view
        # coordinates we also have to get the transform matrix for the
        # current scene view
        comb_trans_mat = get_world_to_view_matrix(f.scene)
        view_coords = \
            apply_transform_to_points(hmgns_world_coords, comb_trans_mat)

        # to get normalized view coordinates, we divide through by the fourth
        # element
        norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

        # the last step is to transform from normalized view coordinates to
        # display coordinates.
        view_to_disp_mat = get_view_to_display_matrix(f.scene)
        disp_coords = apply_transform_to_points(norm_view_coords, view_to_disp_mat)

        # at this point disp_coords is an Nx4 array of homogenous coordinates
        # where X and Y are the pixel coordinates of the X and Y 3D world
        # coordinates, so lets take a screenshot of mlab view and open it
        # with matplotlib so we can check the accuracy
        img = mlab.screenshot()
        pl.imshow(img)

        for i in range(N):
            print('Point %d:  (x, y) ' % i, disp_coords[:, 0:2][i])
            pl.plot([disp_coords[:, 0][i]], [disp_coords[:, 1][i]], 'ro')

        pl.show()

        # you should check that the printed coordinates correspond to the
        # proper points on the screen

        mlab.show()
'''



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')

# plot the unit sphere for reference (optional)
#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = np.outer(np.cos(u), np.sin(v))
#y = np.outer(np.sin(u), np.sin(v))
#z = np.outer(np.ones(np.size(u)), np.cos(v))
#ax.plot_surface(0.9*x, 0.9*y, 0.9*z, color=(0, 0, 0), alpha=0.5)
# plot generator points
#ax.plot(hotSpotCoordinates[:, 0], hotSpotCoordinates[:, 1], hotSpotCoordinates[:, 2], color = (0, 0, 0), marker = '.', linestyle = 'none')
mlab.points3d(hotSpotCoordinates[:, 0], hotSpotCoordinates[:, 1], hotSpotCoordinates[:, 2], scale_factor = 0.1, color = (0, 0, 0))

'''
pts = mlab.points3d(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], scale_factor = 0.01)
mesh = mlab.pipeline.delaunay2d(pts)

#mlab.triangular_mesh(particleCoordinates[:, 0], particleCoordinates[:, 1], particleCoordinates[:, 2], mesh)
#mlab.triangular_mesh(mesh)
surf = mlab.pipeline.surface(mesh)
'''

#print(particleCoordinates[:, 2])
#print(particleCoordinates[:, 2:3])

#print(np.shape(particleCoordinates[:, 2:3]))
#print(np.shape(0.1*np.random.rand(np.size(particleCoordinates, 0), 1)))

#a = 1 + 0.2*np.random.rand(np.size(particleCoordinates, 0), 1)
#particleCoordinates[:, 0:1] *= a
#particleCoordinates[:, 1:2] *= a
#particleCoordinates[:, 2:3] *= a

for iVoronoi in range(numberOfHotSpots):
    a = voronoiID == iVoronoi
    #ax.plot(particleCoordinates[a[:, 0], 0], particleCoordinates[a[:, 0], 1], particleCoordinates[a[:, 0], 2], marker='.',
    #        linestyle='none')

for iVoronoi in range(numberOfHotSpots):
    a = voronoiID == iVoronoi
    x = particleCoordinates[a[:, 0], 0]
    y = particleCoordinates[a[:, 0], 1]
    z = particleCoordinates[a[:, 0], 2]

    #triangles = [(0, i, i + 1) for i in range(1, np.size(x))]
    #x = np.r_[0, x]
    #y = np.r_[0, y]
    #z = np.r_[1, z]
    #print(x[0])
    #print(y[0])
    #print(z[0])
    #print('  ')
    #print('  ')
    randomColour = np.random.rand(1, 3)
    pts = mlab.points3d(x, y, z, color=tuple(randomColour.reshape(1, -1)[0]), scale_factor=0.01)

    xLarger = x[z >= 0]
    yLarger = y[z >= 0]
    zLarger = z[z >= 0]
    xSmaller = x[z < 0]
    ySmaller = y[z < 0]
    zSmaller = z[z < 0]

    #print(zLarger == np.min(zLarger))
    #xMiddle = xLarger[zLarger == np.min(zLarger)]
    #yMiddle = yLarger[zLarger == np.min(zLarger)]
    #zMiddle = zLarger[zLarger == np.min(zLarger)]


    #pts1 = mlab.points3d(xLarger, yLarger, zLarger, color = tuple(randomColour.reshape(1, -1)[0]), scale_factor = 0.01)
    #pts2 = mlab.points3d(xSmaller, ySmaller, zSmaller, color=tuple(randomColour.reshape(1, -1)[0]), scale_factor=0.01)

    # Create and visualize the mesh
    #mesh = mlab.pipeline.elevation_filter(pts)
    #mesh = mlab.pipeline.
    #mesh = mlab.pipeline.delaunay3d(pts)
    mesh1 = mlab.pipeline.delaunay2d(pts)
    #mesh2 = mlab.pipeline.delaunay2d(pts2)
    #surf = mlab.pipeline.surface(mesh, color=tuple(randomColour.reshape(1, -1)[0]))
    surf = mlab.pipeline.surface(mesh1, color = tuple(randomColour.reshape(1, -1)[0]))
    #surf = mlab.pipeline.surface(mesh2, color=tuple(randomColour.reshape(1, -1)[0]))
    # surf = mlab.pipeline.surface(mesh1 + mesh2, color = tuple(randomColour.reshape(1, -1)[0]))




    #mlab.triangular_mesh(x, y, z, triangles)

    #mlab.mesh(x, y, z)
    #mlab.surf(x, y, z)

    #triangles = [(0, i, i + 1) for i in range(1, np.size(x))]
    #triangular_mesh(x, y, z, triangles) # , scalars=t
mlab.show()

quit()
'''
====:====:====:====:====:====:====:====:====:====:====:====:====:====
====:====:====:====:====:====:====:====:====:====:====:====:====:====
====:====:====:====:====:====:====:====:====:====:====:====:====:====
'''

