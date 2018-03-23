#pragma once

#include <map>      // std::map - enum LUT
#include <string>   // std::to_string
#include <iostream> // std::cout
#include <math.h> 	// std::pow, std::exp
#include <sstream>  // std::ostringstream

// Generic debugging
#define DEBUG 0

// Save frames to file
#define DEBUG_SAVEFRAMES 0

// Show FPS counter on top left hand corner of stream
#define DEBUG_FPS 1

#define IMG_ROWS 60
#define IMG_COLS 90

// #define USING_KMEANS
//#define USING_HLS_OPENCV

/*
 * The certainty of a classification is calculated as 1 / sum(results^n)
 * where n determines the spread of the certainties (sensitivity to small changes in results)
 * the larger it is, the smaller non-zero results will get and so the sum reduces => increased classification output
 */
int certainty_spread  = 10;

const std::string BNN_ROOT_DIR  = "/opt/python3.6/lib/python3.6/site-packages/bnn/";
const std::string BNN_LIB_DIR   = "/opt/python3.6/lib/python3.6/site-packages/bnn/src/network/output/sw/";
//const std::string BNN_LIB_DIR = BNN_ROOT_DIR + "libraries";
const std::string BNN_BIT_DIR   = BNN_ROOT_DIR + "bitstreams";
const std::string BNN_PARAM_DIR = BNN_ROOT_DIR + "params/cifar10";
const std::string BNN_PARAM_ORIG_DIR = BNN_ROOT_DIR + "params/cifar10/old";
const std::string USER_DIR      = "/home/xilinx/open_cv2/";

// In order of the steps on which they are applied
enum state {
	RAW,
	GRAY,
	BLUR,
	DIFF,
	THRESH,
	DILATE
};

state curState;
std::map<state, std::string> state2string = {
	{ RAW, "Raw" },
	{ GRAY, "Gray" },
	{ BLUR, "Blur" },
	{ DIFF, "Diff" },
	{ THRESH, "Thresh" },
	{ DILATE, "Dilate" }
};

// OpenCV key codes
enum key : int {
	ONE 	= 49,
	TWO 	= 50,
	THREE 	= 51,
	FOUR 	= 52,
	FIVE 	= 53,
	SIX 	= 54,

	UP 		= 2490368,
	DOWN 	= 2621440,

	RIGHT 	= 2555904,
	LEFT 	= 2424832
};

std::map<key, std::string> key2string = {
	{ ONE,  "One" },
	{ TWO,  "Two" },
	{ THREE,  "Three" },
	{ FOUR,  "Four" },
	{ FIVE,  "Five" },
	{ SIX,  "Six" }
};

std::map<key, state> num2state = {
	{ONE,  RAW},
	{TWO, GRAY},
	{THREE, BLUR},
	{FOUR, DIFF},
	{FIVE, THRESH},
	{SIX, DILATE}
};

enum roi_mode : int {
	BACKGROUND_SUBTRACTION   = 1,
	STRIDE					 = 2,
	IMAGE_SIMPLIFICATION	 = 3,
	CUSTOM_METHOD			 = 4,
	SINGLE_CENTER_BOX		 = 5,
};

template<typename T>
inline void print_vector(std::vector<T> &vec)
{
	std::cout << "{ ";
	for(auto const &elem : vec)
	{
		std::cout << elem << " ";
	}
	std::cout << "}";
}


template<typename T>
inline std::vector<float> normalise(std::vector<T> &vec)
{	
	std::vector<float> cp(vec.begin(), vec.end());
	T mx = *max_element(std::begin(cp), std::end(cp));
	
	for(auto &elem : cp)
		elem = (float)elem / mx;
	
	return cp;
}

template<typename T>
float exp_decay(T lambda, int t, int N = 1)
{
	return N * std::exp(-(lambda * (T)t));
}

template<typename T1, typename T2>
std::vector<T1> scale_vector(std::vector<T1> &in, T2 factor)
{
	std::vector<T1> out;
	
	for(auto const &v : in)
	{
		out.push_back(v * factor);
	}
}

template<typename T>
float calculate_certainty(std::vector<T> &vec)
{
	// Normalise the vector
	std::vector<float> norm_vec = normalise(vec);
	
	float sum = 0;
	for(auto const &elem : norm_vec)
		sum += std::pow(elem, certainty_spread);
	
	if(sum == 0){
		std::cout << "Division by zero, sum = 0" << std::endl;
		return -1;
	}
	
	// Max element = 1 / sum 
	return 1.0 / sum;
}

template<typename T>
unsigned int getMaxIndex(std::vector<T> &container)
{
	T maxVal 			= container[0];
	unsigned int maxInd = 0;
	for(unsigned int i = 1; i < container.size(); i++)
	{
		if(container[i] > maxVal)
		{
			maxInd = i;
			maxVal = container[i];
		}
	}
	
	
	return maxInd;
}

std::string GetUniqueId() 
{
   static int n = 1;
   std::ostringstream os;
   os << n++;
   return os.str();
}

template<typename T>
std::vector<T> getSubset(std::vector<T> &vec, int out_size)
{
	std::vector<T> subset(out_size);
	for(int i = 0; (i < out_size && i < vec.size()); ++i) // Loop up to the limit or the size of the vector (if too small)
		subset[i] = vec[i];
	
	return subset;
}