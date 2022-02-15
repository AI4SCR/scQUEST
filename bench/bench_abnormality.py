"""
1. Train clf
2. Predict epithelial cells
3. Select only epithelial cells
4. Select patients with paired samples
5. Select only cells from N samples, 230'000 cells
5.1. MinMax scale these cells
5. Train AE with this subset of epithelial cells
6. Predict abnormality of epithelial cells
"""

"""
1. Manually labelled Epithelial cells
2. No censoring, asinh co-factor = 5
3. Train clf

"""

"""
1. select epithelial cells
2. downsample 1000 cells per sample 
3. no censoring
4. k=100
"""
