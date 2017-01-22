#!/bin/bash -f
mkdir results
mkdir output
mkdir boundary_data
nvcc -o dbie dbie.cu -arch=sm_20 -use_fast_math -Xptxas -v,-dlcm=cg
echo done....