#!/usr/bin/env sh
# code inspired by Xinyi

SRC_DIR=haptic_src

cd $SRC_DIR
swig -python touch_haptic.i
python setup.py build_ext
cd ..
cp $SRC_DIR/build/lib.*/_touch_haptic.*.so $SRC_DIR
