'''
This file is used to test the storage class.

The user uses a easygui box window to select att .pkl file in the Worlds folder. That file is opened using pickle. The
content of the file is then visualized using mayavi.

'''
import easygui
fileName = easygui.fileopenbox(default = "*.pkl")
if fileName == None:
    quit()
else:
    import pickle  # https://docs.python.org/3/library/pickle.html
    import Visualization
    world = pickle.load(open(fileName, 'rb'))
    #world = pickle.load(open('Worlds/test_save.pkl', 'rb'))


# Creates a mayavi window and visualizes the initial terrain.
window = Visualization.MayaviWindow()
window.Surf(world.initialTotalMap, type='terrain', scene='original')

# Visualizes the eroded terrain.
window.Surf(world.rockMap, type='terrain', scene='updated')
rockMapCopy = world.rockMap.copy()
rockMapCopy[world.sedimentMap == 0] = 0
window.Surf(rockMapCopy + world.sedimentMap, type='sediment', scene='updated')

window.configure_traits()

