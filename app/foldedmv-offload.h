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

#include <cstddef> // fix using ::max_align_t error

#include <string>
#include <iostream>
#include "tiny_cnn/tiny_cnn.h"
#include "ap_int.h"
#include "xblackboxjam_hw.h"
#include "xbackgroundsubtraction_hw.h"
#include "common.h" // ROWS, COLS

//#define HLS_NO_XIL_FPO_LIB
//#include "hls_video.h"
//#ifdef USING_HLS_OPENCV
	//#include <hls_video.h>
	//#include <hls_stream.h>
//#endif

using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord) * 8;
const unsigned int bitsPerUint8_t = sizeof(uint8_t) * 8;

#ifndef VIRTUAL
	#define INPUT_BUF_ENTRIES 3840000
	#define OUTPUT_BUF_ENTRIES 160000
#else
	#define INPUT_BUF_ENTRIES 8192
	#define OUTPUT_BUF_ENTRIES 1024
#endif
#define FOLDEDMV_INPUT_PADCHAR 0

/*#ifndef VIRTUAL
	// Multiply virtual by 300?
	#define CUR_BUF_ENTRIES 6480000
	#define OUTIMG_BUF_ENTRIES 6480000
	//#define DEBUG_BUF_ENTRIES 100
#else*/
	// 720 * 480 = 345600
	// 360 * 240 = 86400
	// 180 * 120 = 21600
	// 90 * 60   = 5400
	// 45 * 30   = 1350
	//#define CUR_BUF_ENTRIES 345600
	//#define PREV_BUF_ENTRIES 345600
	//#define OUTIMG_BUF_ENTRIES 345600
	// 174x144 = 25344
	// 640x480 = 307200
	#define CUR_BUF_ENTRIES 307200
	#define PREV_BUF_ENTRIES 307200
	#define OUTIMG_BUF_ENTRIES 307200
//#endif

void binarizeAndPack(const tiny_cnn::vec_t& in, ExtMemWord* out, unsigned int inBufSize = INPUT_BUF_ENTRIES);

void unpackAndDebinarize(const ExtMemWord* in, tiny_cnn::vec_t& out);

unsigned int paddedSize(unsigned int in, unsigned int padTo);

void FoldedMVOffload(const tiny_cnn::vec_t& in,
    tiny_cnn::vec_t& out,
    unsigned int offloadID,
    tiny_cnn::OffloadConvParams* convParams);

void FoldedMVOffloadBinarized(
    const ExtMemWord* in,
    ExtMemWord* out,
    const unsigned int inBufWords,
    const unsigned int outBufWords,
    const unsigned int numImages);

void FoldedMVInit(const char* attachName);

void FoldedMVDeinit();

void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ExtMemWord val);

void FoldedMVLoadLayerMem(std::string dir, unsigned int peCount, unsigned int layerNo, unsigned int linesWMem, unsigned int linesTMem);

void testPrebinarized(std::vector<tiny_cnn::vec_t>& imgs, std::vector<tiny_cnn::label_t>& labels, const unsigned int labelBits);

std::vector<unsigned int> testPrebinarized_nolabel(std::vector<tiny_cnn::vec_t>& imgs, const unsigned int labelBits, float& usecPerImage);

std::vector<unsigned int> testPrebinarized_nolabel_multiple_images(std::vector<tiny_cnn::vec_t>& imgs, const unsigned int labelBits, float& usecPerImage);

std::string getBNNRoot();

template <typename LowPrecType>
void copyFromLowPrecBuffer(void* buf, tiny_cnn::vec_t& out)
{
	// Convert void pointer (of type LowPrecType) to a vec_t type?
    LowPrecType* lpbuf = (LowPrecType*)buf;
    for (unsigned int i = 0; i < out.size(); i++) {
        out[i] = (tiny_cnn::float_t)lpbuf[i];
    }
}

template <unsigned int inWidth, unsigned int SIMDWidth, typename T>
//void quantiseAndPack(const tiny_cnn::vec_t& in, ExtMemWord* out, unsigned int inBufSize = INPUT_BUF_ENTRIES)
void quantiseAndPack(const T& in, ExtMemWord* out, unsigned int inBufSize = INPUT_BUF_ENTRIES)
{
    if ((in.size() * inWidth) > (inBufSize * bitsPerExtMemWord)) {
        throw "Not enough space in input buffer";
    }
    // first, fill the target buffer with padding data
    memset(out, 0, inBufSize * sizeof(ExtMemWord));
    ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
    // now pack each quantised value as required.
    for (unsigned int i = 0; i < in.size(); i++) {
        ap_fixed<inWidth, 1, AP_TRN, AP_SAT> fxdValue = in[i];
        ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth>*>(&fxdValue); // Interpret the fixed value as an integer.
        ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> bitsPerExtMemWord - inWidth)); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
        out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth * (i % (bitsPerExtMemWord / inWidth)));
    }
}

#include "platform.hpp"
#include <vector>

extern DonutDriver * thePlatform;
extern DonutDriver * thePlatform2;
extern void * accelBufIn, * accelBufOut;
extern ExtMemWord * bufIn, * bufOut;

extern void * accelBufCur, * accelBufPrev, * accelBufOutImg;
//extern hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1> * bufCur, * bufOutImg;
extern uint8_t * bufCur, * bufPrev, * bufOutImg;

void ExecAccel();
void ExecAccel2();

template <unsigned int inWidth, unsigned int SIMDWidth>
void FixedFoldedMVOffload(const tiny_cnn::vec_t& in,
    tiny_cnn::vec_t& out,
    unsigned int offloadID,
    tiny_cnn::OffloadConvParams* convParams)
{
    // always operates on a single image per call for now -- set numImages to 1
    thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, 1); // 0x54
    // binarize input and pack into bit stream
    quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

    // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
    unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
    // copy into accelerator input
    const unsigned int numInpWords = (paddedInDim / (bitsPerExtMemWord / inWidth));
    thePlatform->copyBufferHostToAccel((void*)bufIn, accelBufIn, sizeof(ExtMemWord) * numInpWords);

    // launch
    ExecAccel();

    // TODO add parameters to function call to control how output copy will be done
    if (offloadID == 0xdeadbeef) {
        // TODO make this controllable -- hacked in for cifar10 for 2-byte (nonbinarized activations) now
        unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
        const unsigned int numOutWords = (paddedOutDim / bitsPerExtMemWord);
        thePlatform->copyBufferAccelToHost(accelBufOut, (void*)bufOut, sizeof(ExtMemWord) * numOutWords);
        copyFromLowPrecBuffer<unsigned short>((void*)bufOut, out);
    }
    else {
        // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
        unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

        // copy from accelerator output
        const unsigned int numOutWords = (paddedOutDim / bitsPerExtMemWord);
        thePlatform->copyBufferAccelToHost(accelBufOut, (void*)bufOut, sizeof(ExtMemWord) * numOutWords);

        // unpack output bits and convert output back to float
        unpackAndDebinarize(bufOut, out);
    }
}

template <unsigned int inWidth, unsigned int outWidth>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t>& imgs, std::vector<tiny_cnn::label_t>& labels, const unsigned int numCategories, unsigned int img_num, unsigned int pso2 = 16)
{
    const unsigned int count = img_num; //imgs.size();

//    cout << "Packing and interleaving CIFAR-10 inputs..." << endl;

    // # of ExtMemWords per image
	std::cout << "imgs[0].size() = " << imgs[0].size() << std::endl;
    const unsigned int psi = paddedSize(imgs[0].size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    // # of ExtMemWords per output
    const unsigned int pso = paddedSize(numCategories * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    if (INPUT_BUF_ENTRIES < count * psi)
        throw "Not enough space in accelBufIn";
    if (OUTPUT_BUF_ENTRIES < count * pso)
        throw "Not enough space in accelBufOut";


    cout << "psi size:" << psi << "pso size:" << pso << endl;
//    cout << "pso2 size:" << pso2 << endl;


    // allocate host-side buffers for packed input and outputs
    ExtMemWord* packedImages = new ExtMemWord[(count * psi)];
    ExtMemWord* packedOut = new ExtMemWord[(count * pso2)]; // 16 works

    tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
    // interleave and pack inputs
    for (unsigned int i = 0; i < count; i++) {
        tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
        quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
    }
	

//    cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;

	std::cout << "Before Exec" << std::endl;
    // set number of images to recognize
    thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, count); // 0x54
    // copy inputs to accelerator
    thePlatform->copyBufferHostToAccel((void*)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
    // recognize
    auto t1 = chrono::high_resolution_clock::now();
    ExecAccel();
    auto t2 = chrono::high_resolution_clock::now();
	std::cout << "After Exec" << std::endl;
	
    // copy results back to host
    thePlatform->copyBufferAccelToHost(accelBufOut, (void*)packedOut, sizeof(ExtMemWord) * count * pso2);
    // compare against labels
    unsigned int ok = 0, failed = 0;
    tiny_cnn::vec_t outTest(numCategories, 0);
    for (unsigned int i = 0; i < count; i++) {
        copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso2], outTest);
        unsigned int maxInd = 0;
        unsigned short maxVal = 0;
        for (unsigned int j = 0; j < numCategories; j++) {
            if (outTest[j] > maxVal) {
                maxVal = outTest[j];
                maxInd = j;
            }
        }

        // Debug failed classification
        for (unsigned int j = 0; j < numCategories; j++) {
            cout << "outTest [" << j << "] : " << outTest[j] << endl;
        }
        cout << "Expected: " << labels[i] << " Found: " << maxInd << " MaxVal: " << maxVal << endl;
        cout << "Expected: " << outTest[labels[i]] << " Found: " << outTest[maxInd] << endl;
        cout << endl;

        if (maxInd == labels[i])
            ok++;
        else
            failed++;
    }
	
    cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0 * (float)ok / count << "%" << endl;

    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    float usecPerImage = (float)duration / (count);

    cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
    cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;

    delete[] packedImages;
    delete[] packedOut;
}

template <unsigned int inWidth, unsigned int outWidth>
std::vector<unsigned int> testPrebuiltCIFAR10_from_image(std::vector<tiny_cnn::vec_t>& imgs, const unsigned int numCategories, float& usecPerImage)
{
	//return {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int count = 1;

//    cout << "Packing and interleaving CIFAR-10 inputs..." << endl;

    // # of ExtMemWords per image
    const unsigned int psi = paddedSize(imgs[0].size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    // # of ExtMemWords per output
    const unsigned int pso = paddedSize(64 * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    //const unsigned int pso = 8;
	if (INPUT_BUF_ENTRIES < count * psi)
        throw "Not enough space in accelBufIn";
    if (OUTPUT_BUF_ENTRIES < count * pso)
        throw "Not enough space in accelBufOut";
	
	//std::cout << "old pso = " << psod << ", new pso = " << pso << std::endl;	
	
    // allocate host-side buffers for packed input and outputs
    ExtMemWord* packedImages = new ExtMemWord[(count * psi)];
    ExtMemWord* packedOut = new ExtMemWord[(count * pso)];

    tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
    // interleave and pack inputs
    for (unsigned int i = 0; i < count; i++) {
        tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
        quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
    }

//    cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;

    // copy inputs to accelerator
    thePlatform->copyBufferHostToAccel((void*)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
    // set number of images to recognize
    thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, count); // 0x54
    // recognize
    auto t1 = chrono::high_resolution_clock::now();
    ExecAccel();
    auto t2 = chrono::high_resolution_clock::now();
    // copy results back to host
    thePlatform->copyBufferAccelToHost(accelBufOut, (void*)packedOut, sizeof(ExtMemWord) * count * pso);
    // compare against labels
    unsigned int ok = 0, failed = 0;
    tiny_cnn::vec_t outTest(numCategories, 0);
    copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);
    std::vector<unsigned int> result;
    for (unsigned int j = 0; j < numCategories; j++) {
        result.push_back(outTest[j]);
    }
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    usecPerImage = (float)duration / (count);

    cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
    cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;

    delete[] packedImages;
    delete[] packedOut;
    return (result);
}

template <unsigned int inWidth, unsigned int outWidth>
std::vector<unsigned int> testPrebuiltCIFAR10_multiple_images(std::vector<tiny_cnn::vec_t>& imgs, const unsigned int numCategories, std::vector<unsigned int>& detailed_results, float& usecPerImage, unsigned int pso2 = 16)
{
    const unsigned int count = imgs.size();
    std::vector<unsigned int> results;

//    cout << "Packing and interleaving CIFAR-"
//            "10 inputs..."
//         << endl;

    // # of ExtMemWords per image
    const unsigned int psi = paddedSize(imgs[0].size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    // # of ExtMemWords per output
    const unsigned int pso = paddedSize(64 * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
    if (INPUT_BUF_ENTRIES < count * psi)
        throw "Not enough space in accelBufIn";
    if (OUTPUT_BUF_ENTRIES < count * pso2)
        throw "Not enough space in accelBufOut";
    // allocate host-side buffers for packed input and outputs
    ExtMemWord* packedImages = new ExtMemWord[(count * psi)];
    ExtMemWord* packedOut = new ExtMemWord[(count * pso2)];

    tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32 * 32, false);
    // interleave and pack inputs
    for (unsigned int i = 0; i < count; i++) {
        tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
        quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
    }

//    cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;

    // copy inputs to accelerator
    thePlatform->copyBufferHostToAccel((void*)packedImages, accelBufIn, sizeof(ExtMemWord) * count * psi);
    // set number of images to recognize
    thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, count); // 0x54
	
    // recognize
    auto t1 = chrono::high_resolution_clock::now();
    ExecAccel();
    auto t2 = chrono::high_resolution_clock::now();
    // copy results back to host
    thePlatform->copyBufferAccelToHost(accelBufOut, (void*)packedOut, sizeof(ExtMemWord) * count * pso2);
    tiny_cnn::vec_t outTest(numCategories, 0);
    for (unsigned int i = 0; i < count; i++) {
        copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso2], outTest);
        unsigned int maxInd = 0;
        unsigned short maxVal = 0;
        for (unsigned int j = 0; j < numCategories; j++) {
            detailed_results.push_back(outTest[j]);
            if (outTest[j] > maxVal) {
                maxVal = outTest[j];
                maxInd = j;
            }
        }
        results.push_back(maxInd);
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    usecPerImage = (float)duration / (count);
	
//    cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
//    cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;

    delete[] packedImages;
    delete[] packedOut;
    return (results);
}

//void doImageStuff_acc(hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1> &cur, hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1> &output)
template<unsigned int NROWS, unsigned int NCOLS> // HEIGHT, WIDTH
void doImageStuff_acc(std::vector<uint8_t> &cur, std::vector<uint8_t> &prev, std::vector<uint8_t> &output, unsigned int count)
{	
	std::cout << "Packing inputs..." << std::endl;
    // # of uint8_t per image
    //const unsigned int psi = cur.size(); //paddedSize(input_image.size() * inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	//const unsigned int psi = paddedSize(_count * 8, bitsPerUint8_t) / bitsPerUint8_t;
	const unsigned int psi = NROWS * NCOLS; //cur.size();
    // # of ExtMemWords per output
    const unsigned int pso = psi; //OUTIMG_BUF_ENTRIES-1; //paddedSize(8 * outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
	//const unsigned int psi = 1;
	//const unsigned int pso = 1;
	
	//std::cout << "count = " << count << std::endl;
	
	// IF ERROR THORW CONST CHAR THIS IS WHY!! WOW
	if (CUR_BUF_ENTRIES < count * psi) {
		std::cout << "CUR_BUF_ENTRIES = " << CUR_BUF_ENTRIES << std::endl;
		std::cout << "count * psi = " << count * psi << std::endl;
        throw "Not enough space in accelBufCur";
	}
	if (PREV_BUF_ENTRIES < count * psi) {
		std::cout << "PREV_BUF_ENTRIES = " << PREV_BUF_ENTRIES << std::endl;
		std::cout << "count * psi = " << count * psi << std::endl;
        throw "Not enough space in accelBufPrev";
	}
    if (OUTIMG_BUF_ENTRIES < count * pso) {
		std::cout << "OUTIMG_BUF_ENTRIES = " << OUTIMG_BUF_ENTRIES << std::endl;
		std::cout << "count * pso = " << count * pso << std::endl;
        throw "Not enough space in accelBufOutImg";
	}
	
	std::cout << "psi = " << psi << ", pso = " << pso << ", count = " << count << std::endl;
	
    // Allocate host-side buffers for packed input and outputs
	uint8_t* packedImageCur = new uint8_t[(count * psi)];
	uint8_t* packedImagePrev = new uint8_t[(count * psi)];
	uint8_t* packedOut = new uint8_t[(count * pso)];	
	//ap_uint<1>* packedImage = new ap_uint<1>[(count * psi)];
	//ap_uint<1>* packedOut = new ap_uint<1>[(count * pso)];	
	
	//hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1>* packedImage = new hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1>;
	//hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1>* packedOut = new hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1>;
	
	//quantiseAndPack<inWidth, 1>(input_image, &packedImage[0], psi);

    for (unsigned int i = 0; i < count*psi; i++) {
		packedImagePrev[i] = prev[i];
		packedImageCur[i] = cur[i];
    }	
	
	std::cout << "Finished packing inputs" << std::endl;
	std::cout << "Copying host buffer to accel" << std::endl;
	std::cout << "Size to copy = " << sizeof(uint8_t) * count * psi << std::endl;
	
    // copy inputs to accelerator
    //thePlatform->copyBufferHostToAccel((void*)&cur, accelBufCur, IMG_ROWS * IMG_COLS * 8 * count * psi);
	//thePlatform->copyBufferHostToAccel((void*)packedImage, accelBufCur, sizeof(ap_uint<1>) * count * psi);
	thePlatform2->copyBufferHostToAccel((void*)packedImageCur, accelBufCur, sizeof(uint8_t) * count * psi);
	std::cout << "copied cur" << std::endl;
	thePlatform2->copyBufferHostToAccel((void*)packedImagePrev, accelBufPrev, sizeof(uint8_t) * count * psi);
    
	std::cout << "Finished copying host buffer to accel" << std::endl;
	
	// Enable doImage
	//thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_DOIMAGE_DATA, 1); // 0x74
  	
    // Set number of blocks to dilate
    //thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, count); // 0x54	
	
	std::cout << "ExecAccel..." << std::endl;
	
    auto t1 = chrono::high_resolution_clock::now();
    ExecAccel2();
    auto t2 = chrono::high_resolution_clock::now();
	
	std::cout << "Finished ExecAccel" << std::endl;
	
	//thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_DOIMAGE_DATA, 0); // 0x74
	
    // copy results back to host
    thePlatform2->copyBufferAccelToHost(accelBufOutImg, (void*)packedOut, sizeof(uint8_t) * count * pso);
	//thePlatform->copyBufferAccelToHost(accelBufOutImg, (void*)&output, sizeof(uint8_t) * count * pso);
	
	// Put the packed output into a result vector container
	//std::vector<ExtMemWord> result(pso); // Allocate pso size
    //ExtMemWord* lpbuf = (ExtMemWord*)packedOut;
    //for (unsigned int i = 0; i < result.size(); i++) {
    //    result[i] = (ExtMemWord)lpbuf[i];
    //}
	
	std::cout << "Unpacking results" << std::endl;
	
	//std::vector<uint8_t> result(pso); // Allocate pso size
    uint8_t* lpbuf = (uint8_t*)packedOut;
    for (unsigned int i = 0; i < count*pso; i++) {
        output[i] = (uint8_t)lpbuf[i];
    }
	
	//hls::Mat<ROWS, COLS, HLS_8UC1> result(pso); // Allocate pso size
    //hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1>* lpbuf = packedOut;
    //for (unsigned int i = 0; i < result.size(); i++) {
    //    output = lpbuf[i];
    //}
	
	// Print the results
	/*std::cout << "result = { ";
	for(int i = 0; i < result.size(); i++){
		std::cout << unsigned(result[i]) << " ";
	}
	std::cout << "}" << std::endl;*/
	
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    float usecPerBlockImage = (float)duration / (count);
	
	std::cout << "Took " << usecPerBlockImage << "us to dilate the image " << std::endl;
	
	delete[] packedImageCur;
	delete[] packedImagePrev;
    delete[] packedOut;
	
	//return result;
}

template<unsigned int NROWS, unsigned int NCOLS> // HEIGHT, WIDTH
void doImageStuff_acc_arr(uint8_t * cur, uint8_t * prev, uint8_t* output, unsigned int count)
{
	const unsigned int psi = NROWS * NCOLS;
    const unsigned int pso = psi;
	
	// IF ERROR THORW CONST CHAR THIS IS WHY!! WOW
	if (CUR_BUF_ENTRIES < count * psi) {
		std::cout << "CUR_BUF_ENTRIES = " << CUR_BUF_ENTRIES << std::endl;
		std::cout << "count * psi = " << count * psi << std::endl;
        throw "Not enough space in accelBufCur";
	}
	if (PREV_BUF_ENTRIES < count * psi) {
		std::cout << "PREV_BUF_ENTRIES = " << PREV_BUF_ENTRIES << std::endl;
		std::cout << "count * psi = " << count * psi << std::endl;
        throw "Not enough space in accelBufPrev";
	}
    if (OUTIMG_BUF_ENTRIES < count * pso) {
		std::cout << "OUTIMG_BUF_ENTRIES = " << OUTIMG_BUF_ENTRIES << std::endl;
		std::cout << "count * pso = " << count * pso << std::endl;
        throw "Not enough space in accelBufOutImg";
	}
	
    // copy inputs to accelerator
	thePlatform2->copyBufferHostToAccel((void*)cur, accelBufCur, sizeof(uint8_t) * count * psi);  
	thePlatform2->copyBufferHostToAccel((void*)prev, accelBufPrev, sizeof(uint8_t) * count * psi);  
	// Enable doImage
	//thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_DOIMAGE_DATA, 1); // 0x74
    // Set number of blocks to dilate
    //thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_NUMREPS_DATA, count); // 0x54	
	
    auto t1 = chrono::high_resolution_clock::now();
    ExecAccel2();
    auto t2 = chrono::high_resolution_clock::now();
	

	//thePlatform->writeJamRegAddr(XBLACKBOXJAM_CONTROL_ADDR_DOIMAGE_DATA, 0); // 0x74
	
    // copy results back to host
    thePlatform2->copyBufferAccelToHost(accelBufOutImg, (void*)output, sizeof(uint8_t) * count * pso);
	
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    float usecPerBlockImage = (float)duration / (count);
	
	std::cout << "Took " << usecPerBlockImage << "us to dilate the image " << std::endl;
}