.DEFAULT_GOAL := seamcarver

NVCC=nvcc
NVCCFLAGS=-g --relocatable-device-code true

ALL_HPP_FILES = main.hpp driver.hpp stb_image.h stb_image_write.h image.hpp scexec.hpp matrix.hpp
ALL_CPP_FILES = main.cpp driver.cpp image.cpp seam.cpp scseq.cpp scpar.cpp sccuda.cu # sccuda_tex.cu

seamcarver: $(ALL_HPP_FILES) $(ALL_CPP_FILES)
	$(NVCC) $(NVCCFLAGS) -o seamcarver $(ALL_CPP_FILES) -lboost_program_options -lm

clean:
	rm -f seamcarver
