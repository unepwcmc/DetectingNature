Detecting Nature In Pictures
===============

Building
---------------

1. Add Dataset to the `/data` folder

2. Build with `./build.sh`

3. Run with `./DetectingNature`

Building documentation
---------------

1. Run `doxygen doxyfile`

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

* [15 Scenes Dataset](http://www.cs.illinois.edu/homes/slazebni/research/scene_categories.zip) (91.7MB)
	
* [SUN Database](http://groups.csail.mit.edu/vision/SUN1old/SUN397.tar) (37GB)
