# Project_Genesis

The file WDA_test.py can be used to test the water drop erosion algorithm. The test includes 3D visualization using mayavi. The test also contains basic line-by-line profiling using cProfile. Pickle is used to store content after a run. The file Storage_test.py can be used to open previously created worlds, this is done using Pickle.  

## TO DO

### Water Collection
As drops reach a "dead end" they should disolve into water, filling adjacent tiles and spilling over to nearby tiles until all the water has settled or a "Ledge" has been reached. In that case a new smaller drop should be formed at that location containing the remaining water.

### Drop merging
Small drops should have the capability of merging into larger drops. The larger drops should have a bigger erosional effect and may have different parameters compared to the small drops. This could lower the running time since less drops would be simulated (assuming that a significant amount do merge).

### Weathering
Rock should be able to deteriorate into sediment without the influence of water drops. This weathering process should  be present on the entire world, but the strength may depend on factors like: elevation, slope steepness rock type etc. In the weathering process rock is turned into sediment, this sediment can be washed away by waterdrops or it could form aluvial slopes at the base of mountains.

### Sediment types
As of now only one sediment type exist, there could exist more types for exmaple: gravel > pebble > sand > clay > silt. The Weathering process could for example produce gravel, which in turn could be turned into finer sediment if exposed to the erosional effect of water. The different sediment types should require different strong/big drops in order to carry them. Small drops would for example leave the gravel and transport the sand whilst a large drop could transport both gravel and sand.

### Rock types
Different rocktypes should have different parameters, for example one rock type could be very susceptible to weathering whilst another rocktype could be susceptible to both. The different rocktypes could exist in horizontal planes (layers). Another approach would be to use a 3D version of the 2D algorithm used to generate the initial terrain.

### Interactable visualization
One should be able to toogle on/off things like: rock/sediment colour distinction, drop trajectory density, waterfall density, waterdepth, sedimentdepth etc.

### Statistics module
This module could be used to display/plot statistics and distributions of a selected world. Things to be studied could be: Proportions of sediment/rock/water amounts, distribution of lakes/seas (few large and many small, or all of similar size?), maximum sediment depth, sediment depth distribution, average elevation, average slope steepness etc.

### Noise generation test function.

### Water collection test function.

## Requirements:  
  Python  3.6.6  
  numpy   1.13.1  
  scipy   0.19.1  
  mayavi  4.6.2  
  pyqt    4.11.4  
  easygui 0.98.1
