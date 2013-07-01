cd libs/gmm-fisher
cmake .
make

cd ../vlfeat-0.9.16
make

cd ../yael_v300
./configure.sh
make

cd ../..
mkdir -p build
cd build
mkdir -p ${1:-..}
cmake -DCMAKE_INSTALL_PREFIX:PATH=${1:-..} ../src
make all install
