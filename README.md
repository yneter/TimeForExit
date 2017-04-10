1D GPU version / compile with 

g++ timeforexit.cc -std=c++14 -lboost_system -lboost_timer -lOpenCL -O3 

2D OpenMP CPU Version / compile with 

mpic++ timeforexit_2d_cpu.cc -O3 --fast-math -fopenmp -std=c++11
