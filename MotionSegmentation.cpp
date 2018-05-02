// AUTHOR: Roy Miles (student)

// This is the FPGA accelerator block for the motion segmentation

//#include <hls_video.h>
//#include <hls_stream.h>
#include "ap_int.h"
#include "ap_fixed.h"
#include <hls_video.h>
#include <hls_stream.h>

//#include "ImageSegmentation.h"

#define COLS 176 //640 //176
#define ROWS 144 //480 //144
//#define W 1350
// 176*144 = 25344
// 640*480 = 307200
// 135*90 = 12150
// 90*60  = 5400
// 180*120 = 21600

void doCompute(hls::Mat<ROWS,COLS,HLS_8UC1> &cur_m, hls::Mat<ROWS,COLS,HLS_8UC1> &prev_m, uint8_t * out_img)
{
	// Load into matrices
	//hls::AXIM2Mat<COLS,uint8_t,ROWS,COLS,HLS_8UC1>(cur,cur_m);
	//hls::AXIM2Mat<COLS,uint8_t,ROWS,COLS,HLS_8UC1>(prev,prev_m);

#pragma HLS DATAFLOW
	hls::Mat<ROWS,COLS,HLS_8UC1> blur_cur;
	hls::Mat<ROWS,COLS,HLS_8UC1> blur_prev;

	hls::Mat<ROWS,COLS,HLS_8UC1> diff;
	hls::Mat<ROWS,COLS,HLS_8UC1> thresh;

	hls::Mat<ROWS,COLS,HLS_8UC1> inter0;

	//hls::Mat<ROWS,COLS,HLS_8UC1> dst;
	hls::GaussianBlur<10, 10, HLS_8UC1, HLS_8UC1, ROWS, COLS>(cur_m, blur_cur);
	hls::GaussianBlur<10, 10, HLS_8UC1, HLS_8UC1, ROWS, COLS>(prev_m, blur_prev);

	hls::AbsDiff<ROWS, COLS, HLS_8UC1, HLS_8UC1, HLS_8UC1>(blur_cur, blur_prev, diff);
	hls::Threshold<ROWS, COLS, HLS_8UC1, HLS_8UC1>(diff, thresh, 128, 255, HLS_THRESH_BINARY);

	hls::Dilate(thresh, inter0);
	//hls::Erode(inter0, dst);

	hls::Mat2AXIM<COLS,uint8_t,ROWS,COLS,HLS_8UC1>(inter0, out_img);
}

void Segmentation(uint8_t * cur, uint8_t * prev, uint8_t * out_img)
//void BackgroundSubtraction(uint8_t * cur, uint8_t * prev, uint8_t * out_img)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS INTERFACE m_axi offset=slave depth=25344 port=cur
#pragma HLS INTERFACE s_axilite port=cur bundle=control

#pragma HLS INTERFACE m_axi offset=slave depth=25344 port=prev
#pragma HLS INTERFACE s_axilite port=prev bundle=control

#pragma HLS INTERFACE m_axi offset=slave depth=25344 port=out_img
#pragma HLS INTERFACE s_axilite port=out_img bundle=control


	hls::Mat<ROWS,COLS,HLS_8UC1> cur_m;
	hls::Mat<ROWS,COLS,HLS_8UC1> prev_m;

#pragma HLS STREAM variable=cur_m depth=25344
#pragma HLS STREAM variable=prev_m depth=25344

	hls::AXIM2Mat<COLS,uint8_t,ROWS,COLS,HLS_8UC1>(cur,cur_m);
	hls::AXIM2Mat<COLS,uint8_t,ROWS,COLS,HLS_8UC1>(prev,prev_m);

	doCompute(cur_m, prev_m, out_img);

}
