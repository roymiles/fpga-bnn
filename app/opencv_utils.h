#pragma once

#include <vector>
#include <opencv2/core/core.hpp> // cv::Mat
#include "colours.h"
#include <type_traits>

#include <iostream> // literally for one print

// absdiff, threshold, findContours
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream> // std::ofstream
#include <iterator> // std::ostream_iterator
#include <stdio.h> // std::remove

#include "common.h" // USER_DIR

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/*
#ifdef USING_KMEANS
	#include "kmeans/filtering_algorithm_top.h"
	#include "kmeans/filtering_algorithm_util.h"
	#include "build_kdTree.h"
#endif
*/

// Bnn expects a 32x32 image
#define bnnSize cv::Size(32, 32)
#define blurSize cv::Size(9, 9)

struct video_parameters
{
	int threshold_val;
	int threshold_max;
	int threshold_min;
	int threshold_area; // 300 for camera, this will be different for various videos
	float threshold_certainty;
	
	int blur_kernel_size; // The lower this is, the more smaller features can be extracted (but this can also introduce noise)
	
	// Width and height used for striding and for center box dimensions
	int box_width;
	int box_height;
	int stride_xstep;
	int stride_ystep;
	
	int target_box_count;
	int dilation_val; // Any classification below this threshold is ignored
	
	bool bundle_regions;
	bool adaptive_thresholding;
};

extern video_parameters vp; // Hold all the video/image parameters

#define WINDOW_WIDTH  720 // 180 * 4
#define WINDOW_HEIGHT 480 // 120 * 4
//#define WINDOW_WIDTH  180 // 180 * 4
//#define WINDOW_HEIGHT 120 // 120 * 4
//#define BLOCK_WIDTH   90 //180 // COLS
//#define BLOCK_HEIGHT  60 //120 // ROWS
#define BLOCK_WIDTH 30
#define BLOCK_HEIGHT 20

// Convert output/classification to a colour
// Each class should have a different colour, it makes it easier to distinguish when watching the video
std::vector<cv::Scalar> colourList = {
	COLOURS_blueviolet,
	COLOURS_hotpink,
	COLOURS_papayawhip,
	COLOURS_seagreen,
	COLOURS_teal,
	COLOURS_tomato,
	COLOURS_aliceblue,
	COLOURS_dodgerblue,
	COLOURS_maroon,
	COLOURS_navy
};

/*
 * From a dilated image, contour the boundaries and return the regions of interest
 */
std::vector<cv::Rect> contourRegions(cv::Mat &src);

/*
 * Stride over an image and return the rectangles for each tile
 */
std::vector<cv::Rect> strideImage(cv::Mat &img, int width=200, int height=200, int xStep=100, int yStep=100);

/*
 * Subtract the previous frame from the current frame to remove all background features
 */
cv::Mat backgroundSubtraction(cv::Mat &img, cv::Mat &prevImage);

/*
 * Perform k-means clustering to simply the colour palette of the image
 * This image can then be dilated and contoured to extract the regions of interest
 */
cv::Mat imageSimplification(cv::Mat &src);

// Convert matrix into a vector 
// |1 0 0|
// |0 1 0| -> [1 0 0 0 1 0 0 0 1]
// |0 0 1|
template<typename T>
void flatten_mat(cv::Mat &m, std::vector<T> &v)
{
	if(m.isContinuous()) 
	{
		v.assign(m.datastart, m.dataend);
	} 
	else 
	{
		for (int i = 0; i < m.rows; ++i) 
		{
			v.insert(v.end(), m.ptr<T>(i), m.ptr<T>(i)+m.cols);
		}
	}
}

/*
 * Convert image to cifar10 format (de-interleave)
 * Takes in a .png path and outputs a path to resulting .bin
 * [rgb, rgb, rgb ...] -> [rrrr, gggg, bbbb]
 * @param cleanUp - Delete the input (.png) when finished with it *REMOVED*
 * @param in, bufOut - The string parameters are the image paths
 */
//void image_to_cifar(std::string in, std::string &bufOut, uint8_t _label = 1);

template<typename T1, typename T2>
void image_to_cifar(T1 in, T2 &bufOut, uint8_t _label = 1)
{
	std::cout << "Invalid parameter type" << std::endl;
}

//[32][32][3]
// Convert an opencv Mat to an array in cifar10 format
void image_to_cifar_arr(cv::Mat &img, uint8_t * bufOut, uint8_t _label = 1)
{	
	//uint8_t bufOut[32*32*3 + 1];
	bufOut[0] = _label;
	
	const unsigned int num_rows = 32;
	const unsigned int num_cols = 32;
	
	// Red first i rows, j cols
	for(int i = 0; i < num_rows; i++)
		for(int j = 0; j < num_cols; j++)
			bufOut[i + j*num_rows + 1] = img.at<cv::Vec3b>(i,j)[0];;
		
	// Green
	for(int i = 0; i < num_rows; i++)
		for(int j = 0; j < num_cols; j++)
			bufOut[i + j*num_rows + num_rows*num_cols + 1] = img.at<cv::Vec3b>(i,j)[1];
		
	// Blue	
	for(int i = 0; i < num_rows; i++)
		for(int j = 0; j < num_cols; j++)
			bufOut[i + j*num_rows + 2*num_rows*num_cols + 1] = img.at<cv::Vec3b>(i,j)[2];	
}

// Convert an opencv Mat to a vector in cifar10 format
template<>
void image_to_cifar<cv::Mat, std::vector<uint8_t>>(cv::Mat in, std::vector<uint8_t> &out, uint8_t _label)
{
	cv::resize(in, in, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
	
	cv::Mat bgr[3];   // Destination array
	cv::split(in,bgr);// Split source 
	
	// We write the image into the format used in the Cifar-10 dataset for code compatibility
	std::vector<uint8_t> b(32 * 32);
	std::vector<uint8_t> g(32 * 32);
	std::vector<uint8_t> r(32 * 32);
	flatten_mat(bgr[0], b);
	flatten_mat(bgr[1], g);
	flatten_mat(bgr[2], r);
	std::vector<uint8_t> label = {_label};
	
	//std::vector<uint8_t> out;
	out.insert(out.end(), label.begin(), label.end());
	out.insert(out.end(), r.rbegin(), r.rend());
	out.insert(out.end(), g.rbegin(), g.rend());
	out.insert(out.end(), b.rbegin(), b.rend());
}

template<>
void image_to_cifar<std::string, std::string>(std::string in, std::string &bufOut, uint8_t _label)
{
	bufOut = in + ".bin"; // Rename file extension
    std::ofstream output_file(bufOut);	
	
	// We resize the downloaded image to be 32x32 pixels as expected from the BNN
	cv::Mat img_in = cv::imread(in);
	//cv::Mat img_out(cv::Size(32, 32), CV_8UC3);
	std::vector<uint8_t> img_out;
	
	image_to_cifar(img_in, img_out);
	
    std::copy(img_out.rbegin(), img_out.rend(), std::ostream_iterator<char>(output_file));
}

template<typename T1, typename T2>
void images_to_cifar(std::vector<T1> in, T2 &bufOut, std::vector<uint8_t> _label = {})
{
	std::cout << "Invalid parameter type" << std::endl;
}

//template<> // dont ask
//void images_to_cifar_mat<cv::Mat, std::string>(std::vector<cv::Mat> &in, std::string &bufOut, std::vector<uint8_t> labels)
void images_to_cifar_mat(std::vector<cv::Mat> &in, std::string &bufOut, std::vector<uint8_t> labels = {})
{
	bufOut = USER_DIR + "batch.bin"; // Rename file extension
    std::ofstream output_file(bufOut);
	
	std::vector<uint8_t> out;
	int i = 0;
	
	std::vector<uint8_t> label;
	for(int i = 0; i < in.size(); i++)
		label.push_back(1);
	
	// Prepend the labels
	out.insert(out.end(), label.rbegin(), label.rend());
	
	for(auto const &img_in : in)
	{
		//std::cout << "cols = " << img_in.cols << ", rows = " << img_in.rows << std::endl;
		cv::Mat img_32;
		// We resize the downloaded image to be 32x32 pixels as expected from the BNN
		cv::resize(img_in, img_32, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
		
		cv::Mat bgr[3];   // Destination array
		cv::split(img_32,bgr);// Split source 
		
		// We write the image into the format used in the Cifar-10 dataset for code compatibility
		std::vector<uint8_t> b(32 * 32);
		std::vector<uint8_t> g(32 * 32);
		std::vector<uint8_t> r(32 * 32);
		flatten_mat(bgr[0], b);
		flatten_mat(bgr[1], g);
		flatten_mat(bgr[2], r);
		
		/*std::vector<uint8_t> label;
		if(labels.size() == 0){
			label.push_back(1);
		}else{
			label.push_back(labels[i]);
		}*/
		
		//out.insert(out.end(), label.begin(), label.end()); pretty sure labels should all be at start?
		out.insert(out.end(), r.rbegin(), r.rend());
		out.insert(out.end(), g.rbegin(), g.rend());
		out.insert(out.end(), b.rbegin(), b.rend());
		
		i++;
	}
	
	std::copy(out.rbegin(), out.rend(), std::ostream_iterator<char>(output_file));
}

template<>
void images_to_cifar<std::string, std::string>(std::vector<std::string> in, std::string &bufOut, std::vector<uint8_t> labels)
{
	bufOut = USER_DIR + "batch.bin"; // Rename file extension
    std::ofstream output_file(bufOut);
	
	std::vector<uint8_t> out;
	int i = 0;
	for(auto const &in_path : in)
	{
		// We resize the downloaded image to be 32x32 pixels as expected from the BNN
		cv::Mat img_in = cv::imread(in_path);
		cv::resize(img_in, img_in, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
		cv::Mat img_out(cv::Size(32, 32), CV_8UC3);
		
		cv::Mat bgr[3];   // Destination array
		cv::split(img_in,bgr);// Split source 
		
		// We write the image into the format used in the Cifar-10 dataset for code compatibility
		std::vector<uint8_t> b(32 * 32);
		std::vector<uint8_t> g(32 * 32);
		std::vector<uint8_t> r(32 * 32);
		flatten_mat(bgr[0], b);
		flatten_mat(bgr[1], g);
		flatten_mat(bgr[2], r);
		
		std::vector<uint8_t> label;
		if(labels.size() == 0){
			label.push_back(1);
		}else{
			label.push_back(labels[i]);
		}
		
		//out.insert(out.end(), label.begin(), label.end()); pretty sure labels should all be at start?
		out.insert(out.end(), r.rbegin(), r.rend());
		out.insert(out.end(), g.rbegin(), g.rend());
		out.insert(out.end(), b.rbegin(), b.rend());
		
		i++;
	}
	
	std::copy(out.rbegin(), out.rend(), std::ostream_iterator<char>(output_file));
}


//void image_to_cifar2(std::string in, std::string &bufOut); // This generates slightly different .bin??

//void image_to_cifar(cv::Mat &in, cv::Mat &out, uint8_t _label = 1);
//void images_to_cifar(std::vector<cv::Mat> &in, std::vector<cv::Mat> &out, std::vector<uint8_t> _label = {});

/*
 * Save OpenCV matrix to a .png file (toCifar=false) or a .bin file (toCifar=true) 
 * @param toCifar - If true, interleave the colour channels
 */
std::string saveImage(cv::Mat &img, bool toCifar = true);

/*
 * Merge all overlapping rectangles into a single rectangle
 */
void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes);


// readCIFAR10.cc
// 
// feel free to use this code for ANY purpose
// author : Eric Yuan 
// my blog: http://eric-yuan.me/
void read_batch(std::string filename, std::vector<cv::Mat> &vec, cv::Mat &label);
cv::Mat concatenateMat(std::vector<cv::Mat> &vec);
cv::Mat concatenateMatC(std::vector<cv::Mat> &vec);


bool addGaussianNoise(const cv::Mat mSrc, cv::Mat &mDst, double Mean = 0.0, double StdDev = 10.0);
void occludeImage(cv::Mat curImage, int numColumns);

// NOTE: vector2mat and mat2vector need to be both in the same row|column major
template<int NROWS, int NCOLS, typename T>
void vector2mat(std::vector<T> &src, cv::Mat &dst)
{
	//int NCOLS = (int)src.size() / NROWS;
	//dst.reshape(NCOLS, NROWS);
	
	for(int i = 0; i < NROWS; i++) // i -> y
	{
		for(int j = 0; j < NCOLS; j++) // j -> x
		{
			dst.at<T>(i, j) = src[i*NCOLS + j]; // All columns before moving onto the next row
		}
	}
}

template<int NROWS, int NCOLS, typename T>
void arr2mat(T &src, cv::Mat &dst)
{	
	for(int i = 0; i < NROWS; i++) // i -> y
	{
		for(int j = 0; j < NCOLS; j++) // j -> x
		{
			dst.at<T>(i, j) = src[i*NCOLS + j]; // All columns before moving onto the next row
		}
	}
}

template<int NROWS, int NCOLS, int BWIDTH, int BHEIGHT, typename T>
void blockvec2mat(std::vector<T> &src, cv::Mat &dst)
{	
	// Note: dst.at<T>(y, x)
	int numBlocksX = dst.cols / BWIDTH;
	int numBlocksY = dst.rows / BHEIGHT;
	int numBlocks = numBlocksX * numBlocksY;
	//std::cout << "numBlocks = " << numBlocks << std::endl;
	//std::cout << "Mat size = " << dst.cols << "x" << dst.rows << std::endl;
	//std::cout << "Vec size = " << src.size() << std::endl;
	
	const int blockLength = BWIDTH * BHEIGHT;
	//std::cout << "Block length = " << blockLength << std::endl;
	
	// Every BWIDTH*BHEIGHT element is a block
	for(int ny = 0; ny < numBlocksY; ny++) //y
	{
		
	for(int nx = 0; nx < numBlocksX; nx++) //x
	{
		// (nx * numBlocksX) + ny = n (block number)
		int blockInd = ((ny * numBlocksX) + nx) * blockLength; // 0, BWIDTH*BHEIGHT, 2*BWIDTH*BHEIGHT etc
		
		// Now loop through all pixels in this block and assign it to the mat
		for(int i = 0; i < BHEIGHT; i++) // i -> y    //y
		{
			for(int j = 0; j < BWIDTH; j++) // j -> x   //x
			{	
				dst.at<T>(i + ny*BHEIGHT, j + nx*BWIDTH) = src[blockInd + i*BWIDTH + j]; // (y,x)
			}
		}
	} // Cols
	} // Rows
}


				/*if(blockInd + i*BWIDTH + j > src.size())
				{
					// Error
					std::cout << "src.size() = " << src.size() << std::endl;
					std::cout << "blockInd = " << blockInd << std::endl;
					std::cout << "i = " << i << std::endl;
					std::cout << "j = " << j << std::endl;
					//std::cout << "n = " << n << std::endl;
				}
				
				if(i + ny*BHEIGHT > dst.rows)
				{
					// Error
					std::cout << "dst.rows = " << dst.rows << std::endl;
					std::cout << "blockInd = " << blockInd << std::endl;
					std::cout << "i = " << i << std::endl;
					std::cout << "j = " << j << std::endl;
					std::cout << "ny = " << ny << std::endl;
				}
				
				if(j + nx*BWIDTH > dst.cols)
				{
					// Error
					std::cout << "dst.cols = " << dst.cols << std::endl;
					std::cout << "blockInd = " << blockInd << std::endl;
					std::cout << "i = " << i << std::endl;
					std::cout << "j = " << j << std::endl;
					std::cout << "nx = " << nx << std::endl;
				}*/

// The same as flatten_mat
template<typename T>
void mat2vector(cv::Mat &src, std::vector<T> &dst)
{	
	dst.resize(src.rows * src.cols);
	for(int i = 0; i < src.rows; i++) // i -> y
	{
		for(int j = 0; j < src.cols; j++) // j -> x
		{
			dst[i*src.cols + j] = src.at<T>(i, j);
		}
	}
}

template<typename T>
void mat2arr(cv::Mat &src, T &dst)
{	
	dst.resize(src.rows * src.cols);
	for(int i = 0; i < src.rows; i++) // i -> y
	{
		for(int j = 0; j < src.cols; j++) // j -> x
		{
			dst[i*src.cols + j] = src.at<T>(i, j);
		}
	}
}

template<int BWIDTH, int BHEIGHT, typename T>
void mat2blockvec(cv::Mat &src, std::vector<T> &dst)
{	
	// Loop through the Mat (positions of x's)
	// x--x--x--x--x--x--x--x--x
	// |  |  |  |  |  |  |  |  |
	// x--x--x--x--x--x--x--x--x
	// |  |  |  |  |  |  |  |  |
	// x--x--x--x--x--x--x--x--x
	// |  |  |  |  |  |  |  |  |
	// x--x--x--x--x--x--x--x--x
	// |  |  |  |  |  |  |  |  |
	// x--x--x--x--x--x--x--x--x
	
	int n = 0;
	for(int i = 0; i < src.rows; i += BHEIGHT) //y
	{
		for(int j = 0; j < src.cols; j += BWIDTH) //x
		{
			// Now loop through inside a kernel of size BWIDTH x BHEIGHT (position of .'s)
			// x.............x
			// ...............
			// ...............
			// x.............x
			for(int y = 0; y < BHEIGHT; y++) //y
			{
				for(int x = 0; x < BWIDTH; x++) //x
				{
					dst[n] = src.at<T>(i + y , j + x); // (y, x)
					n++;
				}
			}
			
			
		}
	}
}

/*
inline void doImageStuff_thread(cv::Mat &result, cv::Mat &threshFrame, cv::Rect &roi);

#ifdef USING_KMEANS
// recursively split the kd-tree into P sub-trees (P is parallelism degree)
void recursive_split(uint p,
                    uint n,
                    data_type bnd_lo,
                    data_type bnd_hi,
                    uint *idx,
                    data_type *data_points,
                    uint *i,
                    uint *ofs,
                    node_pointer *root,
                    kdTree_type *heap,
                    kdTree_type *tree_image,
                    node_pointer *tree_image_addr,
                    uint n0,
                    uint k,
                    double std_dev)
{
    if (p==P) {
        printf("Sub-tree %d: %d data points\n",*i,n);
        node_pointer rt = buildkdTree(data_points, idx, n, &bnd_lo, &bnd_hi, *i*HEAP_SIZE/2/P, heap);
        root[*i] = rt;
        uint offset = *ofs;
        readout_tree(true, n0, k, std_dev, rt, heap, offset, tree_image, tree_image_addr);
        *i = *i + 1;
        *ofs = *ofs + 2*n-1;
    } else {
        uint cdim;
        coord_type cval;
        uint n_lo;
        split_bounding_box(data_points, idx, n, &bnd_lo, &bnd_hi, &n_lo, &cdim, &cval);
        // update bounding box
        data_type new_bnd_hi = bnd_hi;
        data_type new_bnd_lo = bnd_lo;
        set_coord_type_vector_item(&new_bnd_hi.value,cval,cdim);
        set_coord_type_vector_item(&new_bnd_lo.value,cval,cdim);

        recursive_split(p*2, n_lo, bnd_lo, new_bnd_hi, idx, data_points,i,ofs,root, heap,tree_image,tree_image_addr,n0,k,std_dev);
        recursive_split(p*2, n-n_lo, new_bnd_lo, bnd_hi, idx+n_lo, data_points,i,ofs,root, heap,tree_image,tree_image_addr,n0,k,std_dev);
    }

}

void kmeans_acc(int clusterCount, int cols, int rows, cv::Mat &src)
{
    const uint n = cols*rows*DATA_DIMENSIONS*32; // 16384
    const uint k = clusterCount;   // 128
    const double std_dev = 0.75; //0.20

    uint *idx = new uint[MAX_DATA_POINTS];
    data_type *data_points = new data_type[MAX_DATA_POINTS];
    uint *cntr_indices = new uint[MAX_CENTRES];
    kdTree_type *heap = new kdTree_type[HEAP_SIZE];
    data_type *initial_centre_positions= new data_type[MAX_CENTRES];
	
	// 
	// Load data points
	for(int z = 0; z < DATA_DIMENSIONS; z++){
		for (int y = 0; y < src.rows; y++){
			for (int x = 0; x < src.cols; x++){
				coord_type b;
				b = (coord_type)src.at<cv::Vec3b>(y, x)[z];
				set_coord_type_vector_item(&data_points[y + x*src.rows].value, b, z);
				//data_points[z + y*3 + x*3*src.rows] = src.at<cv::Vec3b>(y, x)[z];
			}
		}
	}
	
    for (uint i=0;i<n;i++) {
        *(idx+i) = i;
    }
	
	// Load initial centers (random placement)
	for(int z = 0; z < DATA_DIMENSIONS; z++){
		for (int y = 0; y < src.rows; y++){
			for (int x = 0; x < src.cols; x++){
				coord_type b;
				b = (coord_type)(data_points[rand() % MAX_DATA_POINTS].value); // Pick a random point from the data_points
				set_coord_type_vector_item(&initial_centre_positions[y + x*src.rows].value, b, z);
			}
		}
	}
	
    // print initial centres
    printf("Initial centres\n");
    for (uint i=0; i<k; i++) {
        printf("%d: ",i);
        for (uint d=0; d<DATA_DIMENSIONS-1; d++) {
            printf("%d ",get_coord_type_vector_item(initial_centre_positions[i].value, d).to_int());
        }
        printf("%d\n",get_coord_type_vector_item(initial_centre_positions[i].value, DATA_DIMENSIONS-1).to_int());
    }
	
	// compute axis-aligned hyper rectangle enclosing all data points
    data_type bnd_lo, bnd_hi;
    compute_bounding_box(data_points, idx, n, &bnd_lo, &bnd_hi);

    node_pointer root[P];
    kdTree_type *tree_image = new kdTree_type[HEAP_SIZE];
    node_pointer *tree_image_addr = new node_pointer[HEAP_SIZE];
    uint z=0;
    uint ofs=0;
    recursive_split(1, n, bnd_lo, bnd_hi, idx, data_points,&z,&ofs,root,heap,tree_image,tree_image_addr,n,k,std_dev);

	// Call the accelerated kmeans 
	
    data_type clusters_out[MAX_CENTRES];
    coord_type_ext distortion_out[MAX_CENTRES];
	
    // print initial centres
    printf("New centres after clustering\n");
    for (uint i=0; i<k; i++) {
        printf("%d: ",i);
        for (uint d=0; d<DATA_DIMENSIONS-1; d++) {
            printf("%d ",get_coord_type_vector_item(clusters_out[i].value, d).to_int());
        }
        printf("%d\n",get_coord_type_vector_item(clusters_out[i].value, DATA_DIMENSIONS-1).to_int());
    }


    delete[] idx;
    delete[] data_points;
    delete[] initial_centre_positions;
    delete[] cntr_indices;

    delete[] heap;
    delete[] tree_image;
    delete[] tree_image_addr;
}
#endif
*/

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}



