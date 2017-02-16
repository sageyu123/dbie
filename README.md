# DBIE
A GPU parallelized Direct Boundary Integral Equation methond for nonlinear force-free magnetic fields.

## Code
````
dbie.cu: main code of the NLFFF extrapolation.
dbie_make_boundary.pro: main code to create an input boundary file.
dbie_bin2sav.pro: main code to convert the output results from the NLFFF extrapolation to IDL sav file.
````
Detailed explanation can be found in the comments of the main codes.

## Usage
cd to dbie diretory, execute the setenv_dbie.sh to setup the environment.
````
./setenv_dbie.sh
````

Create the input file of boundary data. Run ``dbie_make_boundary.pro`` in IDL.
````
IDL> dbie_make_boundary,bx,by,bz,date,time
````

Compile the source code before running the NLFFF extrapolation
````
nvcc -o dbie dbie.cu -arch=sm_20 -use_fast_math -Xptxas -v,-dlcm=cg
````
Assuming the boundary data (bx,by,bz) observed at the date of yyyy-mm-dd hh:mm:ss,
and the dimension of the data are Ndx*Ndy,
you can run the extrapolation to a height of Ndz layers with follow command.
````
./dbie yyyymmdd hh mm  0  Ndx-1  0  Ndy-1  1  Ndz
````
When the extrapolation completed, a binary containing the result from extrapolation will be saved in **results** folder.
The output 3-Dimension magnetic field data can be retrieved to a IDL sav file with ``dbie_bin2sav.pro``
````
IDL> dbie_bin2sav
````
## Authors
### Original Author
- Sijie Yu ([@sjyu1988](https://github.com/sjyu1988))

### Co-Author
- Yihua Yan(yyh@nao.cas.cn)
