# Project_Genesis

The file WDA_test.py can be used to test the water drop erosion algorithm. The test includes 3D visualization using mayavi. The test also contains basic line-by-line profiling using cProfile. Pickle is used to store content after a run. The file Storage_test.py can be used to open previously created worlds, this is done using Pickle.  

## TO DO

#### Water Collection
As drops reach a "dead end" they should disolve into water, filling adjacent tiles and spilling over to nearby tiles until all the water has settled or a "Ledge" has been reached. In that case a new smaller drop should be formed at that location containing the remaining water.


#### Drop merging
Small drops should have the capability of merging into larger drops. The larger drops should have a bigger erosional effect and may have different parameters compared to the small drops. This could lower the running time since less drops would be simulated (assuming that a significant amount do merge).


#### Weathering
Rock should be able to deteriorate into sediment without the influence of water drops. This weathering process should  be present on the entire world, but the strength may depend on factors like: elevation, slope steepness rock type etc. In the weathering process rock is turned into sediment, this sediment can be washed away by waterdrops or it could form aluvial slopes at the base of mountains.


#### Sediment types
As of now only one sediment type exist, there could exist more types for exmaple: gravel > pebble > sand > clay > silt. The Weathering process could for example produce gravel, which in turn could be turned into finer sediment if exposed to the erosional effect of water. The different sediment types should require different strong/big drops in order to carry them. Small drops would for example leave the gravel and transport the sand whilst a large drop could transport both gravel and sand.


#### Rock types
Different rocktypes should have different parameters, for example one rock type could be very susceptible to weathering whilst another rocktype could be susceptible to both. The different rocktypes could exist in horizontal planes (layers). Another approach would be to use a 3D version of the 2D algorithm used to generate the initial terrain.


#### Interactable visualization
One should be able to toogle on/off things like: rock/sediment colour distinction, drop trajectory density, waterfall density, waterdepth, sedimentdepth etc.


#### Statistics module
This module could be used to display/plot statistics and distributions of a selected world. Things to be studied could be: Proportions of sediment/rock/water amounts, distribution of lakes/seas (few large and many small, or all of similar size?), maximum sediment depth, sediment depth distribution, average elevation, average slope steepness etc.


#### Noise generation test function.


#### Water collection test function.

#### Plate tectonics
Since the worlds created using random-noise as the initial state lack complex global features it would be of benifit to improve the initial state. This could be achieved by implementing a simple/complex system for plate tectonics. This system should give rise to mountain ranges, highlands and lowlands. The important feature is that it should create regions of not entirely random terrain. Examples of features to implement: CRUST FOLDING (this should give rise to rough mountain ranges, which are partly symmetrical. The alpes is an example), CRUST OVERLAP (mountain ranges which are not symmetrical, i.e a steep slope on one side and a less steep slope on the other side. The andes is an example), CRUST OPENING (as two crust separate they expose magma between them. This should give rise to new landmasses via volcanic activities).

An idea to a system: The world consists of a layer of particles. Each particle has an x, y, z and maybe a density/mass component. Each particle is part of a plate. Every particle within a plate is attached to every other (or just adjacent) particle/s in that plate by the means of "springs". Every particle is repelled by particles of other plates, this could be compared to the repulsion of magnets. Each particle is affected by a force from the magma underneath. The magma flow gives rise to "currents" which collectively move plates around. How the magma flow should be calculated is not entirely clear, this could probably be found in an article/book. As the plates move around particles of different plates approach each other. At these areas of collision plates can fold if the density/mass difference between the plates is small, this gives rise to mountains. If the density/mass difference is large the plates may overlap (one goes over the other), this could give rise to mountains or plateus. If two plates were to move apart, magma is revealed. This should give rise to new landmasses via vulcanism. Particles can be destroyed if overlaped by other particles and can be created when magma is exposed. At the end the topography of the particles is used to interpolate the height values of each grid cell. The density of particles could perhaps be used to determine rocktypes??? Perhaps each particle should have rock/sediment type??? One could experiment with alternating between the plate tectonics system and the erosion system, i.e: plate tectonics -> erosion -> plate tectonics -> erosion -> plate tectonics -> erosion, this would introduce sediment into the plate system. Perhaps sediment could be turned into a pourus rock type if buried in the plate tectonics system???

#### Sediment deposition
Needs to be looked over, it may bug ig multiple drops run in parallell.


## Requirements:  
  Python  3.6.6  
  numpy   1.13.1  
  scipy   0.19.1  
  mayavi  4.6.2  
  pyqt    4.11.4  
  easygui 0.98.1
