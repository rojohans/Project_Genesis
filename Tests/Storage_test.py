'''
This code is used to test the storage class.

The user uses a easygui box window to select att .pkl file in the Worlds folder. That file is opened using pickle. The
content of the file is then visualized using mayavi.
'''


'''
=================================================================================================================
             READ HERE IF USING MAC OS X
Storage.GetWorldFromFile() needs to be used before the Visualization module is imported, if not the code crashes on
MAC OS X. This probably has to do with the fact that both easygui and mayavi uses the TK library.
=================================================================================================================
'''
import Storage
# The user selects a world from file to view.
world = Storage.GetWorldFromFile()

import Visualization # See description of potential problem above.


'''
The code below should be contained within a function in the Visualization module.
'''
# Creates a mayavi window and visualizes the initial terrain.
window = Visualization.MayaviWindow()
window.Surf(world.initialTotalMap, type='terrain', scene='original')

# Visualizes the eroded terrain.
window.Surf(world.rockMap, type='terrain', scene='updated')
rockMapCopy = world.rockMap.copy()
rockMapCopy[world.sedimentMap == 0] = 0
window.Surf(rockMapCopy + world.sedimentMap, type='sediment', scene='updated')

window.configure_traits()

