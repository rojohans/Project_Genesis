import pickle # https://docs.python.org/3/library/pickle.html
import Visualization


#import easygui
#easygui.egdemo()
#fileName = easygui.fileopenbox()





#world = pickle.load(open(fileName, 'rb'))
world = pickle.load(open('Worlds/test_save.pkl', 'rb'))




# Creates a mayavi window and visualizes the initial terrain.
window = Visualization.MayaviWindow()
window.Surf(world.initialTotalMap, type='terrain', scene='original')


# Visualizes the eroded terrain.
window.Surf(world.rockMap, type='terrain', scene='updated')
rockMapCopy = world.rockMap.copy()
rockMapCopy[world.sedimentMap == 0] = 0
window.Surf(rockMapCopy + world.sedimentMap, type='sediment', scene='updated')


window.configure_traits()