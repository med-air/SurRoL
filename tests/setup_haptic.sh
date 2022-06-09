#!/usr/bin/env sh
# code inspired by Xinyi

SRC_DIR=SRC

cd $SRC_DIR
swig -python test.i
python setup.py build_ext
cd ..
cp $SRC_DIR/build/lib.*/_test.*.so $SRC_DIR/_test.so
