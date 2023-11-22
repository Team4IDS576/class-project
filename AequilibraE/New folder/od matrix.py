# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:55:54 2023

@author: Pragya Subedi
"""

import numpy as np
import pandas as pd

dem = pd.read_csv("NguyenDemand.csv")
nguyen_demand_df=dem
# Get the maximum zone number to determine the matrix size
max_zone = max(nguyen_demand_df['Origin'].max(), nguyen_demand_df['Destination'].max())

# Create a zero-filled matrix with the size of the maximum zone index
od_matrix = np.zeros((max_zone, max_zone), dtype=int)

# Populate the matrix with OD demand values
for _, row in nguyen_demand_df.iterrows():
    # Subtract 1 from the origin and destination indices to convert to zero-based index
    od_matrix[row['Origin'] - 1, row['Destination'] - 1] = row['OD demand']

print(od_matrix)