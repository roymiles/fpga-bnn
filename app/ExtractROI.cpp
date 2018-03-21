// ExtractROI.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

//#include "cvui.h" // opencv header-only gui
#include <ctime>    // fps counter
#include <string>   // std::to_string
#include <map>      // std::map - enum LUT
#include <fstream>  // std::getline

//#include <stdlib.h>
//#include <dlfcn.h>    // - Download shared object library

#define WINDOW_NAME "OpenCV Test"
#define WINDOW_WIDTH 1520
#define WINDOW_HEIGHT 780

// Needed for folded-mvoffload function definitions
//#define OFFLOAD
//#define RAWHLS

// Bnn expects a 32x32 image
#define bnnSize cv::Size(32, 32)

// External functions in the bnn shared object library
extern "C" {
	void load_parameters(const char* path);
	unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage);
	unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail);
	void free_results(unsigned int * result);
	void deinit();
}

const std::string BNN_ROOT_DIR = "/opt/python3.6/lib/python3.6/site-packages/bnn/";
const std::string BNN_LIB_DIR = "/opt/python3.6/lib/python3.6/site-packages/bnn/src/network/output/sw/";
//const std::string BNN_LIB_DIR = BNN_ROOT_DIR + "libraries";
const std::string BNN_BIT_DIR = BNN_ROOT_DIR + "bitstreams";
const std::string BNN_PARAM_DIR = BNN_ROOT_DIR + "params/cifar10";
const std::string USER_DIR = "/home/xilinx/open_cv/";

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

cv::Mat curFrame, prevFrame, grayFrame, blurFrame, diffFrame, threshFrame, dilateFrame;
// A buffer for the output frame
cv::Mat draw;

std::map<state, cv::Mat*> state2frame = {
	{ RAW, &curFrame },
	{ GRAY, &grayFrame },
	{ BLUR, &blurFrame },
	{ DIFF, &diffFrame },
	{ THRESH, &threshFrame },
	{ DILATE, &dilateFrame }
};

enum key : int {
	ONE = 49,
	TWO = 50,
	THREE = 51,
	FOUR = 52,
	FIVE = 53,
	SIX = 54,

	UP = 2490368,
	DOWN = 2621440,

	RIGHT = 2555904,
	LEFT = 2424832
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

//enum classification : int {
//	BOX = 1,
//	CAR = 2,
//	DOG = 3
//};

//std::map<classification, std::string> classification2string = {
//	{BOX, "Box"},
//	{CAR, "Car"},
//	{DOG, "Dog"}
//};

int streamVideo(std::string videoPath);
void runBNN(cv::Mat &img, int &output, float &certainty);
void keyPressed(int keyID);

cv::Scalar fontColour = cv::Scalar(0, 0, 255, 255);

int threshold_val = 25;
int threshold_max = 255;
int threshold_min = 25;
int threshold_area = 100;
int blur_kernel_size = 9; // The lower this is, the more smaller features can be extracted (but this can introduce noise into the mix)

// USE FOR VIDEO CAMERA TARGETTING ONE PERSON
//#define FIX_ROI_COUNT 1
int target_box_count = 1;
int dilation_val = 7;

int cur_num_boxes, prev_num_boxes;

std::vector<std::string> classes;

void helpMessage(int argc, char** argv)
{
	std::cout << argv[0] << " <mode> <src?>" << std::endl;
	std::cout << "mode = video, pack-interleave, image" << std::endl;
	std::cout << "src = image source" << std::endl;
}

int main(int argc, char** argv)
{
	for(int i = 0; i < argc; i++)
		std::cout << "argv[" << i << "]" << " = " << argv[i] << std::endl;	

	if(argc == 3)
	{
		if (strcmp(argv[1], "video") == 0)
		{
			// Read and stream from a video file
			// argv[1] = "fast-cars.mp4"
			return streamVideo(USER_DIR + argv[2]);
		} 
		else if (strcmp(argv[1], "pack-interleave") == 0) {
			// pack and interleave an image
			std::cout << "Not implemented yet" << std::endl;
			return -1;
		}
		else if (strcmp(argv[1], "image") == 0) {
			// classify an image
			cv::Mat img = cv::imread(argv[2]);
			int output;
			float certainty;

			runBNN(img, output, certainty);
			std::cout << "Output = " << classes[output] << std::endl;

			return 0;
		}
		else {
			helpMessage(argc, argv);
			return -1;
		}
	}else{
		std::cout << "argc = " << argc << std::endl;
		helpMessage(argc, argv);
		return -1;
	}
}

int streamVideo(std::string videoPath)
{
	std::cout << "Loading video \"" << videoPath << "\"" << std::endl;
	cv::setNumThreads(1);
	
	void *pythonswlib;
	unsigned int (*inference)(char const*, unsigned int*, int, float*);
	unsigned int libhandle;

	curState = RAW;
	prev_num_boxes = -1;

	cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cout << "ERROR! Unable to open video" << std::endl;
        return -1;
    }
	
	cv::Size frameSize(WINDOW_WIDTH, WINDOW_HEIGHT);

	std::vector<std::vector<cv::Point>> cnts;

   	// Get a list of all the output classes
	std::ifstream file(BNN_PARAM_DIR + "/classes.txt");
	std::cout << "Opening parameters at: " << (BNN_PARAM_DIR + "/classes.txt") << std::endl;
	std::string str; 
	if (file.is_open())
  	{
		std::cout << "Classes: [";
		while (std::getline(file, str))
		{
			std::cout << str << ", "; 
			classes.push_back(str);
		}
		std::cout << "]" << std::endl;
		file.close();
	}
	else
	{
		std::cout << "Failed to open classes.txt" << std::endl;
		return -1;
	}

	clock_t current_ticks, delta_ticks;
	clock_t fps = 0;
	int frame_num = 0;
	while (true) 
	{
		current_ticks = clock();

		// 480(rows) x 640(cols)
		cap >> curFrame; // Get a new frame from camera

		// If finished reading video file
		if (!curFrame.data)
			break;

		// Resize image
		resize(curFrame, curFrame, frameSize); // Just overwrite the curFrame because no longer need the nonscaled version
		// Convert to gray scale
		cvtColor(curFrame, grayFrame, CV_BGR2GRAY);
		// Apply gaussian blue
		blur(grayFrame, blurFrame, cv::Size(blur_kernel_size, blur_kernel_size));

		// Algorithm uses background subtraction, so needs a previous frame to run
		if (prevFrame.empty()) {
			// Copy the current frame onto the previous frame
			blurFrame.copyTo(prevFrame);
			continue; // Move onto next frame
		}

		// Compute the absolute difference between the current frame and prev frame
		absdiff(prevFrame, blurFrame, diffFrame);
		threshold(diffFrame, threshFrame, threshold_val, 255, cv::THRESH_BINARY);
		
		// Dilate the thresholded image to fill in holes, then find contours on thresholded image
		// This removes the noise (e.g. small bits of white that can be mistaken as ROI)
		dilate(threshFrame, dilateFrame, cv::Mat(), cv::Point(-1, -1), dilation_val);

#ifdef FIX_ROI_COUNT
		if (prev_num_boxes > 0) { // Dont change dilatio
			if (prev_num_boxes > target_box_count) {
				dilation_val++; // Need to dilate more if identifying more boxes than target
			}
			else if (prev_num_boxes > target_box_count) {
				dilation_val--;
			}
		}
#endif

		findContours(dilateFrame.clone(), cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		// The draw frame is dependant on the current viewing state
		draw = *state2frame[curState];

		// Loop over the contours
		cv::Rect br;
		int output;
		float certainty;
		cv::Mat bnnInput(bnnSize, CV_8UC1);

		// int total_area = 0;
		int area;
		for (auto const &cnt : cnts)
		{
			// If the contour is too small, ignore it
			area = contourArea(cnt);
			if (area < threshold_area)
				continue;

			// Compute the bounding box for the contour, draw it on the frame,
			// and update the text
			br = boundingRect(cnt);

			// Resize to 32x32
			resize(curFrame(br), bnnInput, cv::Size(32, 32));
			// Classify the ROI
			runBNN(bnnInput, output, certainty);

			// Draw the rectangle
			if (curState == RAW) {
				rectangle(draw, br, cv::Scalar(0, 0, 255));
			}
			else {
				rectangle(draw, br, cv::Scalar(255, 255, 255));
			}
			// ...and the label from the BNN

			std::string certainty_s = std::to_string(certainty);
			certainty_s.erase(certainty_s.find_last_not_of('0') + 2, std::string::npos); // Remove trailing zeros
			putText(draw, classes[output] + ": " + certainty_s, cv::Point(br.x, br.y), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		
			// total_area += area;
		}

		cur_num_boxes = cnts.size();
		// average approach proved to be very unpredictable (not very useful)
		// if(cur_num_boxes != 0)
		//	threshold_area = total_area / cur_num_boxes;

		// Change the threshold base on the derivative of the numnber of boxes wrt to prev frame
		prev_num_boxes = (prev_num_boxes == -1) ? cur_num_boxes : prev_num_boxes;

		threshold_val += (cur_num_boxes - prev_num_boxes);
		// Bound the threshold value [0, 255]
		threshold_val = (threshold_val > threshold_max) ? threshold_max : threshold_val;
		threshold_val = (threshold_val < threshold_min) ? threshold_min : threshold_val;

		threshold_area += (cur_num_boxes - prev_num_boxes);
		// blur_kernel_size += (cur_num_boxes - prev_num_boxes); dynamically changing the kernel size is a bad idea

		prev_num_boxes = cur_num_boxes;

		// All text is drawn within this box, it helps make it easier to read
		rectangle(draw, cv::Rect(0, 0, 400, 200), cv::Scalar(255, 255, 255), CV_FILLED);
		putText(draw, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		putText(draw, "Viewing mode: " + state2string[curState], cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		putText(draw, "Threshold = " + std::to_string(threshold_val), cv::Point(10, 90), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		putText(draw, "Threshold Area = " + std::to_string(threshold_area), cv::Point(10, 120), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		putText(draw, "Blur kernel size = " + std::to_string(blur_kernel_size), cv::Point(10, 150), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
		putText(draw, "Num boxes: " + std::to_string(cur_num_boxes), cv::Point(10, 180), cv::FONT_HERSHEY_PLAIN, 2, fontColour);

		int keyID = cvWaitKey(1);
		if (keyID != -1)
			keyPressed(keyID);

		fontColour = cv::Scalar(0, 0, 0);

		cv::imshow("Extracting ROI", draw);
		
		//out << curFrame;

		// Copy the current frame onto the previous frame
		blurFrame.copyTo(prevFrame);

		std::cout << "Frame num: " << frame_num << std::endl;
		frame_num++;

		delta_ticks = clock() - current_ticks; // Time in ms to read the frame, process it, and render it on the screen
		if (delta_ticks > 0)
			fps = CLOCKS_PER_SEC / delta_ticks;
	}

	return 0;

}

float* usecPerImage;
void runBNN(cv::Mat &img, int &output, float &certainty)
{
	//output = 0;
	//certainty = 11.1;
	//return;

	// When opening this file, it will overwrite all previous data.
	std::ofstream ofs;
	//const std::string filePath = BNN_ROOT_DIR + "tmp.dat";
	const std::string filePath = USER_DIR + "tmp.dat";
	std::cout << filePath << std::endl;
	//ofs.open(filePath, std::ofstream::out | std::ofstream::app);
	ofs.open(filePath);

  	if (ofs.is_open())
  	{
		std::cout << "Running inference" << std::endl;

		// At the moment, just pick a random classification
		//output = static_cast<classification>(rand() % 3);
		cv::resize(img, img, bnnSize);
		// Create tmp file with the image, this is needed because parse_mnist_images 
		// from tiny dnn expects a path
		// See: image_to_cifar @ https://github.com/Xilinx/BNN-PYNQ/blob/master/bnn/bnn.py 
		cv::Mat bgr[3];
		cv::split(img, bgr); // Split source (channels)
		// 2:red, 1:green, 0:blue
		ofs << (uint8_t)1;
		ofs << bgr[2];
		ofs << bgr[1];
		ofs << bgr[0];

		output = inference(filePath.c_str(), (uint8_t)NULL, classes.size(), usecPerImage);
		std::cout << "Output = " << classes[output] << std::endl;

		ofs.close();

		// unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage);
		certainty = rand() % 100;
	}
	else
	{
		std::cout << "Error opening tmp file" << std::endl;
	}
}

void keyPressed(int keyID)
{
	if (keyID >= ONE && keyID <= SIX) 
	{
		curState = num2state[static_cast<key>(keyID)]; // Change state depending on what button was pressed
	}
	else 
	{
		switch (keyID)
		{
		case UP:
			threshold_val++;
			break;
		case DOWN:
			threshold_val--;
			break;
		case RIGHT:
			threshold_area++;
			break;
		case LEFT:
			threshold_area--;
			break;
		}
	}

	//putText(draw, "Pressed: " + std::to_string(keyID), cv::Point(10, 180), cv::FONT_HERSHEY_PLAIN, 2, fontColour);
}

