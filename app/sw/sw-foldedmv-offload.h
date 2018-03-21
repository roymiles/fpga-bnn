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
 * @file foldedmv-offload.h
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/

#pragma once
#include <string>
#include <iostream>
#include "tiny_cnn/tiny_cnn.h"
#include "ap_int.h"
#include "../sds_lib.h"

using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;


#ifndef VIRTUAL
#define INPUT_BUF_ENTRIES       3840000
#define OUTPUT_BUF_ENTRIES      160000
#else
#define INPUT_BUF_ENTRIES	8192
#define OUTPUT_BUF_ENTRIES	1024
#endif
#define FOLDEDMV_INPUT_PADCHAR  0

void binarizeAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES);

void unpackAndDebinarize(const ExtMemWord * in, tiny_cnn::vec_t &out);

unsigned int paddedSize(unsigned int in, unsigned int padTo);

void FoldedMVOffload(const tiny_cnn::vec_t &in,
                     tiny_cnn::vec_t & out,
                     unsigned int offloadID,
                     tiny_cnn::OffloadConvParams * convParams);

void FoldedMVOffloadBinarized(
                    const ExtMemWord * in,
                     ExtMemWord * out,
                    const unsigned int inBufWords,
                    const unsigned int outBufWords,
                    const unsigned int numImages
                  );

void FoldedMVInit(const char * attachName);

void FoldedMVDeinit();

void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ExtMemWord val);

void FoldedMVLoadLayerMem(std::string dir, unsigned int peCount, unsigned int layerNo, unsigned int linesWMem, unsigned int linesTMem);

void testPrebinarized(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int labelBits);

std::vector<unsigned int> testPrebinarized_nolabel(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int labelBits, float &usecPerImage);

std::vector<unsigned int> testPrebinarized_nolabel_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int labelBits, float &usecPerImage);

std::string getBNNRoot();

template<typename LowPrecType>
void copyFromLowPrecBuffer(void * buf, tiny_cnn::vec_t & out) {
  LowPrecType * lpbuf = (LowPrecType *) buf;
  for(unsigned int i = 0; i < out.size(); i++) {
      out[i] = (tiny_cnn::float_t) lpbuf[i];
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth>
void quantiseAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES) {
  if((in.size() * inWidth) > (inBufSize * bitsPerExtMemWord)) {
    throw "Not enough space in input buffer";
  }
  // first, fill the target buffer with padding data
  memset(out, 0, inBufSize * sizeof(ExtMemWord));
  ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
  // now pack each quantised value as required.
  for(unsigned int i=0; i < in.size(); i++) {
      ap_fixed<inWidth, 1, AP_TRN, AP_SAT> fxdValue = in[i];
      ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth> *>(&fxdValue); // Interpret the fixed value as an integer.
      ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> bitsPerExtMemWord - inWidth)); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
      out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth*(i % (bitsPerExtMemWord / inWidth)));
  }
}

#include "bnn-library.h"

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps);

extern ExtMemWord * bufIn, * bufOut;

template<unsigned int inWidth, unsigned int SIMDWidth>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in,
                        tiny_cnn::vec_t &out,
                        unsigned int offloadID,
                        tiny_cnn::OffloadConvParams * convParams)
{
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, false, 0, 0, 0, 0, 1);

  // unpack output bits and convert output back to float
  if(offloadID == 0xdeadbeef) {
      copyFromLowPrecBuffer<unsigned short>((void *)bufOut, out);
  } else {
      unpackAndDebinarize(bufOut, out);
  }
}


template<unsigned int inWidth, unsigned int outWidth>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories, unsigned int img_num, unsigned int pso2 = 16) {
  const unsigned int count = imgs.size(); //imgs.size();
  //unsigned int img_num = 2; // image number to infer 
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(numCategories*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  
  cout << "psi size:" << psi <<  "pso size:" << pso << " memworkd size: " << sizeof(ExtMemWord) << endl;
  //cout << "pso2 size:" << pso2 << endl;
  
	// allocate host-side buffers for packed input and outputs
	ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
	ExtMemWord * packedOut = new ExtMemWord[(count * pso2)];

	/*ExtMemWord * packedImages_sd = (ExtMemWord *)sds_alloc_non_cacheable((img_num * psi)*sizeof(ExtMemWord));
	if (!packedImages_sd) throw "Failed to allocate host buffer";
	ExtMemWord * packedOut_sd = (ExtMemWord *)sds_alloc_non_cacheable((img_num * pso2)*sizeof(ExtMemWord));
	if (!packedOut_sd) throw "Failed to allocate host buffer";  
	cout << "Memory allocated " << endl; */

	tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32*32, false);
	// interleave and pack inputs
	for(unsigned int i = 0; i < count; i++) {
		tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
		quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
	}

	// in data
	//memcpy ((ap_uint<64> *)packedImages_sd, (ap_uint<64> *)packedImages, img_num*psi*sizeof(ExtMemWord)); 
	//memcpy ((ap_uint<64> *)packedImages_sd, (ap_uint<64> *)packedImages, img_num*psi*sizeof(ExtMemWord)); 

	cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
	auto t1 = chrono::high_resolution_clock::now();

	BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count);

	auto t2 = chrono::high_resolution_clock::now();
	cout << "Computation finished" << endl;
	// compare against labels

	// out data
	//memcpy ((ap_uint<64> *)packedOut, (ap_uint<64> *)packedOut_sd, img_num*pso*sizeof(ExtMemWord)); 

  extern unsigned int ok,failed;
  float diff_err,diff_ok; 

  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso2], outTest);
    unsigned int maxInd = 0;
    unsigned short maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i])
      ok++;
    else
      failed++;
  }
  
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/img_num << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / img_num;
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  cout << " diff ok " << (float)diff_ok/(float)(3*ok) << " diff error " << (float)diff_err/(float)(3*failed) << endl; 
  //sds_free(packedImages);
  //sds_free(packedOut);
  //free(packedImages_all);
  //free(packedOut_all);
  
  delete [] packedImages;
  delete [] packedOut;
}


template<unsigned int inWidth, unsigned int outWidth>
std::vector<unsigned int>  testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, float &usecPerImage) {
  const unsigned int count = 1;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32*32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
	copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
  std::vector<unsigned int> result;
	for(unsigned int j = 0; j < numCategories; j++) 
  {
		result.push_back(outTest[j]);
	}
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
  return (result);
}

template<unsigned int inWidth, unsigned int outWidth>
std::vector<unsigned int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t> & imgs, const unsigned int numCategories, std::vector<unsigned int> & detailed_results, float & usecPerImage) {
  const unsigned int count = imgs.size();
  std::vector<unsigned int> results;
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(64*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32*32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  // copy inputs to accelerator
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();
  // compare against labels

  tiny_cnn::vec_t outTest(numCategories, 0);
  
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso], outTest);
    unsigned int maxInd = 0;
    unsigned short maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
    detailed_results.push_back(outTest[j]);
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
	results.push_back(maxInd);
  }  
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
  return (results);
}