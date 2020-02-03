from mayavi import mlab
import pickle
import Root_Directory

def CreateColormap(colormapName):
    '''
    Retrieves the (255, 3) colormap lut table from a mayavi predefined colormap and saves it to file.
    :param colormapName:
    :return:
    '''
    f = mlab.figure()
    tmpObj = mlab.points3d(0, 0, 0, colormap=colormapName)
    lut = tmpObj.module_manager.scalar_lut_manager.lut.table.to_array()
    colormap = lut[:, 0:3]

    fileName = Root_Directory.Path() + '/Templates/Colormap/' + str(colormapName) + '.pkl'
    fileToOpen = open(fileName, 'wb')
    pickle.dump(colormap, fileToOpen, pickle.HIGHEST_PROTOCOL)
    fileToOpen.close()

def LoadColormap(colormapName):
    '''
    Given the name of a saved colormap it is loaded and returned to the user.
    :param colormapName:
    :return:
    '''
    try:
        fileName = Root_Directory.Path() + '/Templates/Colormap/' + str(colormapName) + '.pkl'
        import gc
        gc.disable()
        fileToRead = open(fileName, 'rb')
        colormap = pickle.load(fileToRead)
        fileToRead.close()
        gc.enable()
        return colormap
    except:
        print('The specific colormap is not available, the file might not exist')
        quit()