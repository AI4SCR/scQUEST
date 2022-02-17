#!/bin/bash

#conda activate starProtocols
#
python benchmark.py 'torch' 10000 small 10 256 10
python benchmark.py 'torch' 10000 large 10 256 10
python benchmark.py 'light' 10000 small 10 256 10
python benchmark.py 'light' 10000 large 10 256 10
#
python benchmark.py 'torch' 10000 small 250 256 10
python benchmark.py 'torch' 10000 large 250 256 10
python benchmark.py 'light' 10000 small 250 256 10
python benchmark.py 'light' 10000 large 250 256 10
#
python benchmark.py 'torch' 1000000 small 10 256 10
python benchmark.py 'torch' 1000000 large 10 256 10
python benchmark.py 'light' 1000000 small 10 256 10
python benchmark.py 'light' 1000000 large 10 256 10

