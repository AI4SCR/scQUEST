#!/bin/bash

#conda activate scQUEST
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py torch torch
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py large_torch torch
#
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py keras keras
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py large_keras keras
#
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py light light_default
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py light light_noCheckNoLog
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py light light_AccPrec
#
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py large_light light_default
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py large_light light_noCheckNoLog
python /Users/art/Documents/scQUEST/bench/benchmark_lightning.py large_light light_AccPrec
