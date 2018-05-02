// This file has been modified (by roy miles, student) to add driver support for the other accelerated blocks
// Author: Xilinx, Modified by Me

/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file main_python.c
 *
 * Host code for BNN, overlay CNV-Pynq, to manage parameter loading, 
 * classification (inference) of single and multiple images
 * 
 *
 *****************************************************************************/
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <algorithm>

#include "opencv_utils.h" // image_to_cifar_arr

#include "common.h" // calculate_certainty
using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

unsigned int ok,failed;
ofstream myfile;
  

void makeNetwork(network<mse, adagrad> & nn) {
  nn
      << chaninterleave_layer<identity>(3, 32*32, false)
      << offloaded_layer(3*32*32, 10, &FixedFoldedMVOffload<8, 1>, 0xdeadbeef, 0)
      ;
}

extern "C" void init_test()
{
	std::cout << "FoldedMVInit..." << std::endl;
	FoldedMVInit("cnv-pynq");
	std::cout << "Finished FoldedMVInit" << std::endl;
	
	network<mse, adagrad> nn;
	makeNetwork(nn);
}


extern "C" void load_parameters(const char* path)
{
	#include "config.h"
	
	std::cout << "FoldedMVInit..." << std::endl;
	FoldedMVInit("cnv-pynq");
	std::cout << "Finished FoldedMVInit" << std::endl;
	
	network<mse, adagrad> nn;
	makeNetwork(nn);

	cout << "Setting network weights and thresholds in accelerator..." << endl;
	
	FoldedMVLoadLayerMem(path , 0, L0_PE, L0_WMEM, L0_TMEM);
	FoldedMVLoadLayerMem(path , 1, L1_PE, L1_WMEM, L1_TMEM);
	FoldedMVLoadLayerMem(path , 2, L2_PE, L2_WMEM, L2_TMEM);
	FoldedMVLoadLayerMem(path , 3, L3_PE, L3_WMEM, L3_TMEM);
	FoldedMVLoadLayerMem(path , 4, L4_PE, L4_WMEM, L4_TMEM);
	FoldedMVLoadLayerMem(path , 5, L5_PE, L5_WMEM, L5_TMEM);
	FoldedMVLoadLayerMem(path , 6, L6_PE, L6_WMEM, L6_TMEM);
	FoldedMVLoadLayerMem(path , 7, L7_PE, L7_WMEM, L7_TMEM);
	FoldedMVLoadLayerMem(path , 8, L8_PE, L8_WMEM, L8_TMEM);
	
	cout << "Finished setting network weights" << endl;
}

// Custom method that directly takes the cv::Mat rather than a path (accepts in cifar10 format)
//[32][32][3]
/*extern "C" unsigned int inference_arr(uint8_t * img_in, unsigned int results[64], int number_class, float *usecPerImage)
{
	FoldedMVInit("cnv-pynq");
	
	network<mse, adagrad> nn;

	const unsigned int num_rows = 32;
	const unsigned int num_cols = 32;	
	
	makeNetwork(nn);
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images(num_rows);
	
	for(int i = 0; i < num_rows; i++)
	{
		vec_t d = {};
		for(int j = 0; j < num_cols; j++)
		{
			d.push_back(img_in[i + j*num_rows]);
		}
		
		test_images[i] = d;
	}
	
	std::vector<unsigned int> class_result;
	float usecPerImage_int;
	class_result=testPrebuiltCIFAR10_from_image<8, 16>(test_images, number_class, usecPerImage_int);
}*/

extern "C" unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage)
{
	//FoldedMVInit("cnv-pynq");
	//network<mse, adagrad> nn;
	//makeNetwork(nn);
	
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;

	parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
	std::vector<unsigned int> class_result;
	float usecPerImage_int;
	class_result=testPrebuiltCIFAR10_from_image<8, 16>(test_images, number_class, usecPerImage_int);
	
	//std::cout << "class_results[" << class_result.size() << "] = { ";
	/*for(int i = 0; i < class_result.size(); i++)
	{
		if(i != 0)
			std::cout << ", ";
		
		std::cout << class_result[i];
	}
	std::cout << std::endl;*/
	//std::cout << "}" << std::endl;	
	
	if(results)
		std::copy(class_result.begin(),class_result.end(), results);
	if (usecPerImage)
		*usecPerImage = usecPerImage_int;
	
	return (std::distance(class_result.begin(), std::max_element(class_result.begin(), class_result.end())));
}

//extern "C" std::vector<uint8_t> doImageStuff_test(std::vector<uint8_t> &frames)
//extern "C" hls::Mat<ROWS, COLS, HLS_8UC1> doImageStuff_test(hls::Mat<ROWS, COLS, HLS_8UC1> &frames, hls::Mat<ROWS, COLS, HLS_8UC1> &output)
extern "C" void doImageStuff_test(std::vector<uint8_t> &cur, std::vector<uint8_t> &prev, std::vector<uint8_t> &output, unsigned int count)
{
	// Needs to be done to ensure buffers are allocated
	/*std::cout << "FoldedMVInit..." << std::endl;
	FoldedMVInit("cnv-pynq");
	std::cout << "Finished" << std::endl;*/
	
	//std::cout << "Performing image acceleration using a block size of " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << std::endl;
	//doImageStuff_acc<BLOCK_HEIGHT, BLOCK_WIDTH>(cur, prev, output, count);
	doImageStuff_acc<WINDOW_HEIGHT, WINDOW_WIDTH>(cur, prev, output, count);
}

extern "C" void doImageStuff_test_arr(uint8_t * cur, uint8_t * prev, uint8_t * output, unsigned int count)
{
	doImageStuff_acc_arr<BLOCK_HEIGHT, BLOCK_WIDTH>(cur, prev, output, count);
}

extern "C" unsigned int inference_test(const char* path, unsigned int results[64], int number_class, float *usecPerImage, unsigned int img_num, int pso2)
{
	std::cout << "FoldedMVInit..." << std::endl;
	FoldedMVInit("cnv-pynq");
	std::cout << "Finished" << std::endl;

	network<mse, adagrad> nn;

	makeNetwork(nn);
	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;

	parse_cifar10(path, &test_images, &test_labels, -1.0, 1.0, 0, 0);
	float usecPerImage_int;

	cout << "--------- <8 , " << pso2 << "> --------" << endl;
	cout << "Running testPrebuiltCIFAR10..." << std::endl;
	testPrebuiltCIFAR10<8, 16>(test_images, test_labels, number_class, img_num, pso2);
	cout << "Finished testPrebuiltCIFAR10" << std::endl;
	
	return 0;
}

extern "C" unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail = 0)
{
	//FoldedMVInit("cnv-pynq");
	//network<mse, adagrad> nn;
	//makeNetwork(nn);

	std::vector<label_t> test_labels;
	std::vector<vec_t> test_images;

	parse_cifar10(path,&test_images, &test_labels, -1.0, 1.0, 0, 0);
	std::vector<unsigned int> all_result;
	std::vector<unsigned int> detailed_results;
	float usecPerImage_int;
	all_result=testPrebuiltCIFAR10_multiple_images<8, 16>(test_images, number_class, detailed_results, usecPerImage_int);
	unsigned int * result;
	if (image_number)
	   *image_number = all_result.size();
	if (usecPerImage)
		*usecPerImage = usecPerImage_int;
	if (enable_detail)
	{
		result = new unsigned int [detailed_results.size()];
		std::copy(detailed_results.begin(),detailed_results.end(), result);
	}
	else
	{
		result = new unsigned int [all_result.size()];
		std::copy(all_result.begin(),all_result.end(), result);
	}
	   
	return result;
}

extern "C" void free_results(unsigned int * result)
{
	delete[] result;
}

extern "C" void deinit() {
	FoldedMVDeinit();
}