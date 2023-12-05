all:
	/lusr/cuda-11.6/bin/nvcc -O3 -o main.o -c main.cpp -lboost_program_options
	/lusr/cuda-11.6/bin/nvcc -O3 -o kmeans_kernel.o  -c kmeans_kernel.cu 
	/lusr/cuda-11.6/bin/nvcc -O3 -o kmeans kmeans.o kmeans_kernel.o -lboost_program_options