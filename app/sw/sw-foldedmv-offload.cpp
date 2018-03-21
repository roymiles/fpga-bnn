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
 * @file sw-foldedmv-offload.cpp
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/
#include "sw-foldedmv-offload.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace tiny_cnn;
using namespace std;

//#define INCLUDE_TRANSFER_TIMES_IN_BENCHMARK

#if defined(INCLUDE_TRANSFER_TIMES_IN_BENCHMARK) || defined(RAWHLS)
#define TRANSFER_EXCL(x) ;
#define TRANSFER_INCL(x) x;
#else
#define TRANSFER_EXCL(x) x;
#define TRANSFER_INCL(x) ;
#endif


string getBNNRoot() {
  char * bnnRoot = getenv ("XILINX_BNN_ROOT");
  if(!bnnRoot)
    throw "XILINX_BNN_ROOT must be set";
  return string(bnnRoot);
}

// return in padded to a multiple of padTo
unsigned int paddedSize(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}

// binarize an array of floating point values according to their sign and
// pack into a stream of bits
void binarizeAndPack(const vec_t & in, ExtMemWord * out, unsigned int inBufSize) {
  if(in.size() / bitsPerExtMemWord > inBufSize)
    throw "Not enough space in input buffer";
  // first, fill the target buffer with padding data
  memset(out, FOLDEDMV_INPUT_PADCHAR, inBufSize * sizeof(ExtMemWord));
  // now clear/set each bit position as needed
  for(unsigned int i=0; i < in.size(); i++) {
      if(in[i] >= 0) {
          // set bit
          out[i / bitsPerExtMemWord] |= ((ExtMemWord)1 << (i % bitsPerExtMemWord));
      } else {
          // clear bit
          out[i / bitsPerExtMemWord] &= ~((ExtMemWord)1 << (i % bitsPerExtMemWord));
      }
  }
}

// unpack a stream of bit and debinarize them into -1 and +1 floating point
// values (where a 0 bit is -1 and 1 bit is +1)
void unpackAndDebinarize(const ExtMemWord * in, vec_t &out) {
  for(unsigned int i=0; i < out.size(); i++) {
      if((in[i / bitsPerExtMemWord] >> (i % bitsPerExtMemWord)) & 0x1) {
          out[i] = 1;
      } else {
          out[i] = -1;
        }
    }
}

void FoldedMVLoadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem)
{
  for(unsigned int pe = 0; pe < peCount; pe++) {
    // load weights
    ifstream wf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-weights.bin", ios::binary | ios::in);
    if(!wf.is_open())
      throw "Could not open file";
    for(unsigned int line = 0 ; line < linesWMem; line++) {
      ExtMemWord e = 0;
      wf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo*2, pe, line, e);
    }
    wf.close();
    // load thresholds
    ifstream tf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-thres.bin", ios::binary | ios::in);
    if(!tf.is_open())
      throw "Could not open file";
    for(unsigned int line = 0 ; line < linesTMem; line++) {
      ExtMemWord e = 0;
      tf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo*2 + 1, pe, line, e);
    }
    tf.close();
  }
}