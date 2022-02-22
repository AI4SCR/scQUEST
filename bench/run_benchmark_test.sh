#!/bin/bash

#conda activate scQUEST
python benchmark.py 'torch' 1000 small 1 256 1
python benchmark.py 'torch' 1000 large 1 256 1
python benchmark.py 'light' 1000 small 1 256 1
python benchmark.py 'light' 1000 large 1 256 1
#
python benchmark.py 'torch' 1000 small 1 256 1
python benchmark.py 'torch' 1000 large 1 256 1
python benchmark.py 'light' 1000 small 1 256 1
python benchmark.py 'light' 1000 large 1 256 1

# rsync -Pav -e "ssh -i ~/.ssh/id-zc2-benchmark" ubuntu@9.4.131.69:/from/dir/ /to/dir/