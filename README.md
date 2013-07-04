Detecting Nature In Pictures
===============

Dependencies
---------------

For the framework:

* [CImg](http://cimg.sourceforge.net/) >= 1.5.4

* [Boost](http://www.boost.org/) >= 1.50.0

* [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) >= 3.12

* [LIBLINEAR](http://www.csie.ntu.edu.tw/~cjlin/liblinear/) >= 1.8

For the Ruby wrapper:

* [Ruby](http://www.ruby-lang.org/) >= 1.9.1

* [SWIG](http://www.swig.org/) >= 2.0.0

Building command-line application
---------------

1. Build with `./build.sh`

2. Run with `./DetectingNature`

Building Ruby Gem
---------------

1. Create Gem with `gem build detectingnature.gemspec`

2. Install with `gem install DetectingNature-X.X.X.gem`

Building documentation
---------------

1. Run `doxygen`

2. Output is saved to the `./docs` folder

3. If the desired output is a PDF, run `make` inside `./docs/latex`

Dataset structure
---------------

	data/
		dataset_name/
			class_name/
				image_name.jpg
				another_image.png
				...
			another_class/
				...
		another_dataset/
			...

		
Example datasets
---------------

* [8 Scenes Dataset](http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip) (129MB)

* [15 Scenes Dataset](http://www.cs.illinois.edu/homes/slazebni/research/scene_categories.zip) (92MB)
	
* [SUN Database](http://groups.csail.mit.edu/vision/SUN1old/SUN397.tar) (37GB)

* [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) (126MB)

* [Vogel and Schiele](http://www.d2.mpi-inf.mpg.de/sites/default/files/images.zip) (214MB)
