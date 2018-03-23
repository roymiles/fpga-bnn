// test.cpp : Defines the entry point for the console application.
//

#ifdef USING_KMEANS
	#include <cstddef> // fix using ::max_align_t error
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
//#include <sstream> // std::stringstream

#include <ctime>    // fps counter
#include <string>   // std::to_string
#include <fstream>  // std::getline

// Write tmp cifar file from png
#include <fstream>
#include <iterator>     // std::ostream_iterator
#include <algorithm>    // std::copy

#include <stdlib.h>     /* srand, rand */

#include "tiny_cnn/tiny_cnn.h" // tiny_cnn::vec_t
//#include "bnn-library.h" // BlackBoxJam
//#include "foldedmv-offload.h" // ExtMemWord
//using namespace hls;

#include "app.h"
#include "common.h"
#include "opencv_utils.h"
#include "args.hxx"

#define HLS_NO_XIL_FPO_LIB
//#ifdef USING_HLS_OPENCV
	//#include <hls_opencv.h>
	//#include <hls_video.h>
	//#include <hls_stream.h>
	//#include "hls/hls_test.h"
//#endif

#define WINDOW_NAME "OpenCV Test"
#define LINE_THICKNESS 4

// External functions that are either defined in a linked library or compiled driver source file
// This is dependant on the compilation method.
extern "C" {
	void load_parameters(const char* path);
	unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage);
	unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail);
	void free_results(unsigned int * result);
	void deinit();

	unsigned int inference_test(const char* path, unsigned int results[64], int number_class, float *usecPerImage, unsigned int img_num, int pso);
	
	//std::vector<uint8_t> doImageStuff_test(std::vector<uint8_t> &frames);
	//void doImageStuff_test(hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1> &frames, hls::Mat<IMG_ROWS, IMG_COLS, HLS_8UC1> &output);
	void doImageStuff_test(std::vector<uint8_t> &frames, std::vector<uint8_t> &output, unsigned int count);
	void doImageStuff_test_arr(uint8_t* &frames, uint8_t* &output, unsigned int count); // Avoids allocating host buffers every single time...
	
	unsigned int inference_arr(uint8_t * img_in, unsigned int results[64], int number_class, float *usecPerImage);
}

// How many previous classifications should be remembered
#define HST_LEN 32
float lambda = 0.2; // Decay constant
std::vector<float> history_weights(HST_LEN);
std::vector<std::vector<unsigned int> > result_history;

struct scene_object
{
	cv::Rect pos; // Most recent position
	std::vector<cv::Point> prev_points;
	std::vector<std::vector<unsigned int> > result_history; // All its previous classifications
};

std::vector<scene_object> scene_objects;

int num_results = 0;
int roi_limit 	= -1; // Limit the number of classifications per frame to this number. If -1, then no limit
// Run BNN on every region in a current frame, and? draw results onto the frame image
void classifyRegions(cv::Mat &curFrame, std::vector<cv::Rect> &regions, std::vector<std::string> &classes, std::vector<unsigned int> &outputs, std::vector<float> &certainties, bool bundleRegions, std::vector<std::vector<unsigned int> > &detailed_results)
{
	cv::Mat rct;
	unsigned int output;
	float certainty;

	if(bundleRegions)
	{
		// Parse all the regions into the BNN at once
		std::vector<cv::Mat> rois;
		for(auto &r : regions)
			rois.push_back(curFrame(r)); // Get the rectangle from the frame
		
		runBNN_multipleImages(rois, classes, outputs, certainties, detailed_results);	
	}
	else
	{
		std::cout << "classifying " << regions.size() << " images" << std::endl;
		// Classify each region one at a time
		int i = 0;
		for(auto const &region : regions)
		{			
			// Classify the ROI
			rct = curFrame(region);
			
			// Save image to tmp file
			std::string tmp_path = USER_DIR + "tmp.png";
			cv::imwrite(tmp_path, rct);
			
			std::string out_path;
			image_to_cifar(tmp_path, out_path); // Convert png to cifar10 format (to_cifar2 does not work)
			
			// Delete (clean up) the .png input file
			if(std::remove(tmp_path.c_str()) != 0)
				std::cout << "Failed to delete tmp file: " << tmp_path << std::endl;
			
			// Check detailed_results works for multiple regions of interest this way
			runBNN_image(out_path.c_str(), classes, output, certainty, detailed_results[i]);
			
			// Delete .bin file
			if(std::remove(out_path.c_str()) != 0)
				std::cout << "Failed to delete tmp .bin file: " << out_path << std::endl;				
			
			// Maybe resize these before the for loop?
			outputs[i] 	   = output;
			certainties[i] = certainty;
			
			i++;
			num_results++;
		}
	
	}
}

void testBNN(const char* batch_path, int &output, float &certainty, int num_images, int pso)
{
	unsigned int results[64];
	int number_class = 10;
	float* usecPerImage;

    inference_test(batch_path, results, number_class, usecPerImage, num_images, pso);
	
	return;
}

void runBNN_multipleImages(std::vector<cv::Mat> &imgs, std::vector<std::string> &classes, std::vector<unsigned int> &outputs, std::vector<float> &certainties, std::vector<std::vector<unsigned int> > &detailed_results)
{	
	// Save all the images (cv::Mat) to temp files
	/*std::vector<std::string> tmp_paths(imgs.size());
	std::string tmp;
	for(int i = 0; i < imgs.size(); i++)
	{
		tmp = USER_DIR + "tmp_" + std::to_string(i) + ".png";
		tmp_paths[i] = tmp;
		cv::imwrite(tmp, imgs[i]);
	}*/
	
	//unsigned int results[64];
	int number_class = 10; // Number of classes (birds, dogs etc...)
	float* usecPerImage;
	int* imageNumber;
	
	// Show the regions
	//int i = 0;
	//for(auto const &img : imgs){
	//	cv::imshow("ROI img_" + std::to_string(i), img);
	//	i++;
	//}
	
	unsigned int pso = 16; // Should be program argument
	unsigned int enable_detail = 1;
	
	unsigned int* results;
	
	// Merge all the images into one .bin file
	std::string batch_path;
	images_to_cifar_mat(imgs, batch_path); // Pass in the vector of mats directly		
	
	// unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail = 0)
	results = inference_multiple(batch_path.c_str(), number_class, imageNumber, usecPerImage, enable_detail);
	
	// Get the results and the certainty
	std::vector<unsigned int> r(number_class); // Stores all the results in a single array
	unsigned int tmp; // Don't ask  - WHY DO I NEED THIS??
	for(int i = 0; i < imgs.size(); i++)
	{
		// Overwrite previous
		for(int j = 0; j < number_class; j++) {
			detailed_results[i][j] = results[i*number_class + j];
			r[j] = results[i*number_class + j];
		}
		
		//std::cout << "r = "; print_vector(r); std::cout << std::endl;
		tmp = getMaxIndex(r);
		//std::cout << "max index = " << tmp << std::endl;
		//std::cout << "certainty = " << calculate_certainty(r) << std::endl;
		// Pass the results to output vector
		outputs[i] 	   = tmp;
		certainties[i] = calculate_certainty(r);
	}
	
	//std::cout << "outputs = "; print_vector(outputs);
	//std::cout << "certainties = "; print_vector(certainties);
	
	// Delete (clear up) all temp files (.png)
	//for(auto const &tmp_path : tmp_paths)
	//	if(std::remove(tmp_path.c_str()) == 0)
	//		std::cout << "Failed to delete tmp file: " << tmp_path << std::endl;
	
	return;
}

// Running BNN on an image (return index result)
// Output = index (classification)
// Certainty = float, square difference from max value and all the other classifications
void runBNN_image(const char* path, std::vector<std::string> &classes, unsigned int &output, float &certainty, std::vector<unsigned int> &detailed_results)
{
	//unsigned int results[64]; // Allows up to 64 classes - maybe make it classes.size?
	//int number_class = 10; // Remove magic numbers?
	int number_class = classes.size();
	unsigned int results[number_class];
	float* usecPerImage;
	
	// unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage);
	output = inference(path, results, number_class, usecPerImage);
	
	//std::vector<unsigned int> truncated_results(number_class); // Only look at the results for the classes. Not the excess results padding
	for(int i = 0; i < number_class; i++)
	{
		//truncated_results[i] = results[i];
		detailed_results[i] = results[i];
	}
	
	//std::cout << "truncated_results = "; print_vector(truncated_results); std::cout << std::endl;
	//std::cout << "detailed_results = "; print_vector(detailed_results); std::cout << std::endl;

	certainty = calculate_certainty(detailed_results);
	
	//std::cout << "------------------" << std::endl;
	//std::cout << "Certainty = " << certainty << std::endl;
	//std::cout << "Distance = " << output << std::endl; // the classification
	
	return;
}

int main(int argc, char** argv)
{   
	args::ArgumentParser parser("Program description...", "");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	args::ValueFlag<std::string> mode_arg(parser, "camera|image|video|test|cifar10|create_batch|open_batch|occlusion_test|noise_test", "What to run...", {"mode"});
	args::ValueFlag<std::string> src_arg(parser, "path", "src to .mp4|.png|.bin file depending on mode", {"src"});
	
	args::ValueFlag<int> roi_method_arg(parser, "BG_SUB=1|STRIDE=2|IMG_SIMPLE=3|CUSTOM_METHOD=4|SINGLE_CENTER_BOX=5", "Method of extracting regions of interest", {"roi_method"});
	args::ValueFlag<bool> save_video_arg(parser, "bool", "Save the output to a video file", {"save_video"});
	
	args::ValueFlag<int> num_images_arg(parser, "integer", "Number of images to process. Used in test mode only", {"num_images"});
	args::ValueFlag<int> pso_arg(parser, "integer", "Size of pso. Used in test mode only", {"pso"});
	
	args::ValueFlag<float> threshold_certainty_arg(parser, "float", "The threshold certainty for successful classification", {"threshold_certainty"});
	args::ValueFlag<int> threshold_area_arg(parser, "int", "The threshold area for extracted regions of interest", {"threshold_area"});
	
	args::ValueFlag<int> box_width_arg(parser, "int", "Width of the stride rectangle/center box", {"box_width"});
	args::ValueFlag<int> box_height_arg(parser, "int", "Height of the stride rectangle/center box", {"box_height"});
	args::ValueFlag<int> stride_xstep_arg(parser, "int", "x step", {"stride_xstep"});
	args::ValueFlag<int> stride_ystep_arg(parser, "int", "y step", {"stride_ystep"});
	
	args::ValueFlag<int> dilation_val_arg(parser, "int", "Dilation coefficient", {"dilation_val"});
	args::ValueFlag<int> threshold_val_arg(parser, "int", "Threshold value", {"threshold_val"});
	
	args::ValueFlag<int> batch_num_arg(parser, "int", "What image to pull from the batch", {"batch_num"});
	args::ValueFlag<int> bundle_regions_arg(parser, "bool", "Enable/disable region bundling. This drastically increasing the throughput by calling the BNN in a single call and avoiding unnecessary I/O operations", {"bundle_regions"});
	
	args::ValueFlag<int> adaptive_thresholding_arg(parser, "bool", "Enable/disable adaptive thresholding. If disabled, a hard threshold will be used", {"adaptive_thresholding"});
	
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
	
	// Pre-populate history weights with exponential decays
	for(int i = 0; i < HST_LEN; i++)
	{
		history_weights[i] = exp_decay(lambda, i);
	}
	
	// Because push_back reads from opposite direction
	// std::reverse(history_weights.begin(), history_weights.end()); 
	
	std::cout << "history weights = "; print_vector(history_weights); std::cout << std::endl;
	
	if (mode_arg) 
	{                                                                                                                                       
		std::string mode_s = args::get(mode_arg);
		std::cout << "Mode: " << mode_s << std::endl; 
		
		// Extract all the global arguments
		vp.dilation_val = (dilation_val_arg ? static_cast<int>(args::get(dilation_val_arg)) : 8);
		vp.threshold_val = (threshold_val_arg ? static_cast<int>(args::get(threshold_val_arg)) : 70);
		vp.threshold_certainty = (threshold_certainty_arg ? static_cast<float>(args::get(threshold_certainty_arg)) : 0.8);
		vp.threshold_area = (threshold_area_arg ? static_cast<int>(args::get(threshold_area_arg)) : 500);
		vp.bundle_regions = (bundle_regions_arg ? static_cast<bool>(args::get(bundle_regions_arg)) : true);
		vp.adaptive_thresholding = (adaptive_thresholding_arg ? static_cast<bool>(args::get(adaptive_thresholding_arg)) : true);
		
		std::cout << "Dilation val: " << vp.dilation_val << std::endl;
		std::cout << "Threshold val: " << vp.threshold_val << std::endl;
		std::cout << "Threshold certainty: " << vp.threshold_certainty << std::endl;
		std::cout << "Threshold area: " << vp.threshold_area << std::endl;

		std::cout << "Adaptive thresholding set to " << vp.adaptive_thresholding << std::endl;

		// Used in camera|video
		roi_mode mode = BACKGROUND_SUBTRACTION;
		if(roi_method_arg)
			mode = static_cast<roi_mode>(args::get(roi_method_arg));		
		
		// Used in camera|video
		if(mode == roi_mode::STRIDE || mode == roi_mode::SINGLE_CENTER_BOX)
		{
			// Global variables found in opencv_utils.h
			vp.box_width = (box_width_arg ? static_cast<int>(args::get(box_width_arg)) : 200);
			vp.box_height = (box_height_arg ? static_cast<int>(args::get(box_height_arg)) : 200);
			vp.stride_xstep = (stride_xstep_arg ? static_cast<int>(args::get(stride_xstep_arg)) : 100);
			vp.stride_ystep = (stride_ystep_arg ? static_cast<int>(args::get(stride_ystep_arg)) : 100);
			
			std::cout << "Box width: " << vp.box_width << std::endl;
			std::cout << "Box height: " << vp.box_height << std::endl;
			std::cout << "Stride x step: " << vp.stride_xstep << std::endl;
			std::cout << "Stride y step: " << vp.stride_ystep << std::endl;
		}
		
		
		if(mode_s == "camera")
		{
			std::cout << "--- RUNNING CAMERA ---" << std::endl;
			
			std::vector<std::string> classes = loadClasses();

			bool save_video = (save_video_arg ? static_cast<bool>(args::get(save_video_arg)) : false);

			std::cout << "roi_method: " << mode << std::endl;
			std::cout << "save_video: " << save_video << std::endl;
			
			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());
			
			streamVideo(mode, classes, "", save_video); // If no src supplied, the camera is used
		}
		else if(mode_s == "image")
		{
			std::cout << "--- RUNNING IMAGE ---" << std::endl;
			
			// Assign a default value and overwrite it if the user supplies the parameter
			std::string src_s = (src_arg ? args::get(src_arg) : "/home/xilinx/open_cv2/images/custom/cars/1.jpg");
			std::cout << "src: " << src_s << std::endl;
		
			unsigned int output;
			float certainty;
			
			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());

			// If first character is no a forward slash, then user is specifying a relative path
			// pre-concatinate the absolute path to the program
			if(src_s[0] != '/')
				src_s = "/home/xilinx/open_cv2/" + src_s;
			
			cv::Mat image = cv::imread(src_s);
			cv::imshow("Image", image);

			std::vector<std::string> classes = loadClasses();
			
			// ------ Using files ------ //
			// Save image to tmp file
			std::string tmp_path = USER_DIR + "tmp.png";
			cv::imwrite(tmp_path, image);
			
			std::string out_path;
			image_to_cifar(tmp_path, out_path); // Convert png to cifar10 format
			
			// Delete (clean up) the .png input file
			if(std::remove(tmp_path.c_str()) != 0)
				std::cout << "Failed to delete tmp file: " << tmp_path << std::endl;
			
			std::vector<unsigned int> detailed_results(classes.size());
			runBNN_image(out_path.c_str(), classes, output, certainty, detailed_results);
			
			// ----- Not using files ---- //

/*
			// Resize the mat
			cv::resize(image, image, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
			
			// Convert to cifar10 format
			uint8_t img_arr_out[32 * 32 * 3];
			image_to_cifar(image.data, img_arr_out);
			
			runBNN_image_arr(img_arr_out, classes, output, certainty);
*/

			// ------------------------- //
			
			// Delete .bin file
			if(std::remove(out_path.c_str()) != 0)
				std::cout << "Failed to delete tmp .bin file: " << out_path << std::endl;			

			std::cout << "Classified as a " << classes[output] << std::endl;				
			
			cv::waitKey(0); // Wait until user enters a key on the console
			
		}
		else if(mode_s == "video")
		{
			std::cout << "--- RUNNING VIDEO ---" << std::endl;
			
			std::string src_s = (src_arg ? args::get(src_arg) : USER_DIR + "videos/street.mp4");			
			bool save_video = (save_video_arg ? static_cast<bool>(args::get(save_video_arg)) : false);

			std::cout << "src: " << src_s << std::endl;
			std::cout << "roi_method: " << mode << std::endl;
			std::cout << "save_video: " << save_video << std::endl;
			
			std::vector<std::string> classes = loadClasses();
			
			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());
					
			streamVideo(mode, classes, src_s, save_video);
		}
		else if(mode_s == "test") // Test the accelerator (the bnn and the image segmentation)
		{
			std::cout << "--- RUNNING TESTS ---" << std::endl;
			
			
			int output;
			float certainty;

			// Extract the arguments or use default values
			int num_images_i = (num_images_arg ? args::get(num_images_arg) : 9);
			int pso_i = (pso_arg ? args::get(pso_arg) : 16);
			std::string src_s = (src_arg ? args::get(src_arg) : "/home/xilinx/host/bnn_lib_tests/test/data_batch_1.bin");
			
			std::cout << "num_images: " << num_images_i << std::endl;
			std::cout << "pso: " << pso_i << std::endl;
			std::cout << "src: " << src_s << std::endl;	

			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());			

			// Test the bnn on the test batches
			testBNN(src_s.c_str(), output, certainty, num_images_i, pso_i);
			
			
// ----------------------------------------------------------------------------------------------------
			// Test the accelerated background subtraction, thresholding and dilation
			std::cout << "Doing image stuff..." << std::endl;
			
			// Load an image and threshold it
			std::string image_path = "/home/xilinx/open_cv2/images/BlackWhiteTest2.jpg";
			//std::string image_path = (src_arg ? args::get(src_arg) : "/home/xilinx/open_cv2/images/BlackWhiteTest2.jpg");
			cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
			
			cv::Size frameSize(WINDOW_WIDTH, WINDOW_HEIGHT);
			resize(image, image, frameSize);
			cv::Mat img_bw = image > 128; // Convert to black and white
			
			cv::imshow("B/W Input: ", img_bw);
			
			// Extract a 90x60 region from the image (for cur and prev)
			cv::Rect roi(0, 0, BLOCK_WIDTH, BLOCK_HEIGHT); // x,y,w,h
			
			// Create N blocks (ROIs) to be dilated, each of size 90x60
			std::vector<cv::Mat> ROIs;
			for(int y = 0; y < WINDOW_HEIGHT; y += BLOCK_HEIGHT)
			{
				for(int x = 0; x < WINDOW_WIDTH; x += BLOCK_WIDTH)
				{
					// Shift the window
					roi.x = x;
					roi.y = y;
					
					cv::Mat ROI = img_bw(roi);
					ROIs.push_back(ROI);
				}
			}
			
			cv::Mat image_bw_roi(BLOCK_HEIGHT, BLOCK_WIDTH, CV_8UC1); // rows,cols,type
			cv::Mat image_roi(BLOCK_HEIGHT, BLOCK_WIDTH, CV_8UC1);
			image_roi  = image(roi); // Extract region from frame
			cv::Mat img_bw_roi = image_roi > 128;	
			
			// ENABLES BUFCUR ETC
			//std::cout << "FoldedMVInit..." << std::endl;
			//FoldedMVInit("cnv-pynq");
			//std::cout << "Finished" << std::endl;
			
			std::cout << "Block dimensions = " << BLOCK_WIDTH << "x" << BLOCK_HEIGHT << std::endl;
			std::cout << "Window dimensions = " << WINDOW_WIDTH << "x" << WINDOW_HEIGHT << std::endl;
			std::cout << "# Regions = " << ROIs.size() << std::endl;
			
			/*
			// NOT PIPELINED HARDWARE
			auto t1 = std::chrono::high_resolution_clock::now();
			for(int i = 0; i < ROIs.size(); i++)
			{
				auto t1a = std::chrono::high_resolution_clock::now();
				// Convert image matrix to a vector (flatten)
				std::vector<uint8_t> acc_input(BLOCK_HEIGHT * BLOCK_WIDTH);
				//hls::Mat<BLOCK_HEIGHT, BLOCK_WIDTH, HLS_8UC1> acc_input_mat; // <ROWS, COLS, TYPE>
				mat2vector(ROIs[i], acc_input); // Convert the 90x60 Mat to vector
				
				// Write the input to the hls mat stream
				//for(int i = 0; i < BLOCK_HEIGHT * BLOCK_WIDTH; i++) {
				//	hls::Scalar<1, uint8_t> b;
				//	b.val[0] = acc_input[i];
				//	acc_input_mat << b;
				//}
				
				std::vector<uint8_t> acc_output(BLOCK_HEIGHT * BLOCK_WIDTH);
				doImageStuff_test(acc_input, acc_output, 1);
				
				//hls::Mat<BLOCK_HEIGHT, BLOCK_WIDTH, HLS_8UC1> acc_output_mat;
				//doImageStuff_test(acc_input_mat, acc_output_mat);
				
				// Convert output vector back into a matrix
				cv::Mat image_out(BLOCK_HEIGHT, BLOCK_WIDTH, CV_8UC1);
				vector2mat<BLOCK_HEIGHT, BLOCK_WIDTH>(acc_output, image_out);
				
				// Write hls mat output back to opencv mat
				//for(int i = 0; i < image_out.rows; i++){
				//	for(int j = 0; j < image_out.cols; j++){
				//		hls::Scalar<1, uint8_t> b;
				//		acc_output_mat >> b; // readout
				//		image_out.at<uint8_t>(i, j) = b.val[0];
				//	}
				//}
				auto t2a = std::chrono::high_resolution_clock::now();
				
				cv::imshow("in", ROIs[i]);
				cv::imshow("out", image_out);
				cv::waitKey(0);	
				
				auto duration1a = std::chrono::duration_cast<std::chrono::microseconds>(t2a - t1a).count();
				std::cout << "[Not-pipelined] ----- Dilation [hardware] on SINGLE images took " << duration1a << "us" << std::endl;
			}	
			auto t2 = std::chrono::high_resolution_clock::now();
			
			auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			std::cout << "[Not-pipelined] Dilation [hardware] on ALL images took " << duration1 << "us" << std::endl;
			*/
			
			// SOFTWARE NOT PIPELINED
			auto t3 = std::chrono::high_resolution_clock::now();
			std::vector<cv::Mat> dilatedFrames;
			for(int i = 0; i < ROIs.size(); i++)
			{
				auto t3a = std::chrono::high_resolution_clock::now();
				// Calculate the difference between frames
				//std::cout << "cur[" << cur.rows << "," << cur.cols << "], prev[" << prev.rows << "," << prev.cols << "]" << std::endl;
				
				//cv::Mat diff = abs(image(roi_cur) - image(roi_prev));
				
				// Threshold the difference frame
				//cv::Mat img_bw = diff > 70;
		
				cv::Mat dilateFrame;
				dilate(img_bw_roi, dilateFrame, cv::Mat(), cv::Point(-1, -1), vp.dilation_val);
				
				dilatedFrames.push_back(dilateFrame);
				
				//cv::Mat dilateFrame;
				//cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
				//cv::morphologyEx(img_bw_roi, dilateFrame, cv::MORPH_CLOSE, structuringElement);
				
				auto t4a = std::chrono::high_resolution_clock::now();
				
				auto duration2a = std::chrono::duration_cast<std::chrono::microseconds>(t4a - t3a).count();
				std::cout << "[Not-pipelined] ----- Dilation [software] on SINGLE images took " << duration2a << "us" << std::endl;
			}	
			auto t4 = std::chrono::high_resolution_clock::now();
			
			auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
			std::cout << "[Not-pipelined] Dilation [software] on ALL images took " << duration2 << "us" << std::endl;	
			
			// SOFTWARE PIPELINED
			auto t5 = std::chrono::high_resolution_clock::now();
			{
				cv::Mat dilateFrameFull;
				dilate(img_bw, dilateFrameFull, cv::Mat(), cv::Point(-1, -1), vp.dilation_val);
				
				cv::imshow("B/W Output[software]: ", dilateFrameFull);
			}
			auto t6 = std::chrono::high_resolution_clock::now();
			
			auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
			std::cout << "[Pipelined] Dilation [software] on ALL images took " << duration3 << "us" << std::endl;
			
			// CONVERSION TEST WORKS
			
				//uint8_t* acc_input_full  = new uint8_t[(WINDOW_HEIGHT * WINDOW_WIDTH)];
				//uint8_t* acc_output_full = new uint8_t[(WINDOW_HEIGHT * WINDOW_WIDTH)];
				std::vector<uint8_t> acc_vec_full(WINDOW_HEIGHT * WINDOW_WIDTH);
				//template<int BWIDTH, int BHEIGHT, typename T>
				mat2blockvec<BLOCK_WIDTH, BLOCK_HEIGHT>(img_bw, acc_vec_full);
				
				//std::cout << "Converted to vec" << std::endl;
				//std::cout << "acc_vec_full = "; print_vector(acc_vec_full); std::cout << std::endl;
				
				cv::Mat image_out(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1); // nrows, ncols, type
				// template<int NROWS, int NCOLS, int BWIDTH, int BHEIGHT, typename T>
				blockvec2mat<WINDOW_HEIGHT, WINDOW_WIDTH, BLOCK_WIDTH, BLOCK_HEIGHT>(acc_vec_full, image_out);					
				
				//std::cout << "Converted back to mat" << std::endl;
				
				cv::imshow("Testing block array conversions", image_out);
				
				//delete[] acc_input_full;
				//delete[] acc_output_full;
				
				cv::waitKey(0);
			
			// END CONVERSION TEST

			// HARDWARE PIPELINED
			auto t7 = std::chrono::high_resolution_clock::now();
			{
				//mat2vector(img_bw, acc_input_full); // Convert the 90x60 Mat to vector	
				
				//uint8_t* acc_input_full  = new uint8_t[(WINDOW_HEIGHT * WINDOW_WIDTH)];
				//uint8_t* acc_output_full = new uint8_t[(WINDOW_HEIGHT * WINDOW_WIDTH)];
				std::vector<uint8_t> acc_input_full(WINDOW_HEIGHT * WINDOW_WIDTH);
				std::vector<uint8_t> acc_output_full(WINDOW_HEIGHT * WINDOW_WIDTH);
				//mat2arr(img_bw, acc_input_full); // Convert the 90x60 Mat to vector
				mat2blockvec<BLOCK_WIDTH, BLOCK_HEIGHT>(img_bw, acc_input_full);
				std::cout << "Converted" << std::endl;
				
				auto t7a = std::chrono::high_resolution_clock::now();
				//std::vector<uint8_t> acc_output_full(WINDOW_HEIGHT * WINDOW_WIDTH);
				doImageStuff_test(acc_input_full, acc_output_full, ROIs.size());
				auto t8a = std::chrono::high_resolution_clock::now();
				
				cv::Mat image_out(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
				//arr2mat<WINDOW_HEIGHT, WINDOW_WIDTH>(acc_output_full, image_out);
				blockvec2mat<WINDOW_HEIGHT, WINDOW_WIDTH, BLOCK_WIDTH, BLOCK_HEIGHT>(acc_output_full, image_out);
				
				// Convert output vector back into a matrix
				//cv::Mat image_out(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
				//vector2mat<WINDOW_HEIGHT, WINDOW_WIDTH>(acc_output_full, image_out);
				
				auto duration4a = std::chrono::duration_cast<std::chrono::microseconds>(t8a - t7a).count();
				std::cout << "[Pipelined] [LOWER-BOUND] Dilation [hardware] on ALL images took " << duration4a << "us" << std::endl;					
				
				cv::imshow("B/W Output[hardware]: ", image_out);
				
				//delete[] acc_input_full;
				//delete[] acc_output_full;
			}
			auto t8 = std::chrono::high_resolution_clock::now();	

			auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count();
			std::cout << "[Pipelined] Dilation [hardware] on ALL images took " << duration4 << "us" << std::endl;				
			
			// Test vector2mat and mat2vector works
			//cv::Mat test(BLOCK_HEIGHT, BLOCK_WIDTH, CV_8UC1);
			//vector2mat<BLOCK_HEIGHT, BLOCK_WIDTH>(dilation_input, test);
			//imshow("Test", test);
			
			std::cout << "Finished dilation" << std::endl;
// ----------------------------------------------------------------------------------------------------
		
			cv::waitKey(0);	
		} 
		else if(mode_s == "open_batch") // Open a binary batch file and show the images inside it
		{
			std::string filename;
			filename = USER_DIR + "batch.bin";;
			std::vector<cv::Mat> batch1;
			cv::Mat label1 = cv::Mat::zeros(1, 10000, CV_64FC1);   
			read_batch(filename, batch1, label1);
			
			std::cout << "Number of images = " << batch1.size() << std::endl;
			std::cout << "Label1 = " << label1.at<int>(0, 0) << std::endl;
			
			for(auto &img : batch1)
				imshow("title", img);	

			cv::waitKey(0);			
		}
		else if(mode_s == "cifar10") // Test that the cifar10 conversion works
		{
			int batch_num = (batch_num_arg ? static_cast<int>(args::get(batch_num_arg)) : 0); // Index of image to check from the batch file			 
			 
			std::string filename = "/home/xilinx/host/bnn_lib_tests/test/data_batch_1.bin";
			std::vector<cv::Mat> batch;
			cv::Mat label1 = cv::Mat::zeros(1, 10000, CV_64FC1);   
			read_batch(filename, batch, label1);
			
			std::cout << "Label1 = " << label1.at<int>(0, 0) << std::endl;
			std::cout << "Number of images = " << batch.size() << std::endl;
			
			std::string original_image, savedName2, savedName3;
			original_image = "batch_image_example.png"; // This will be the reference image for all other conversions
			imwrite(USER_DIR + original_image, batch[batch_num]);
			imshow(original_image, batch[batch_num]);

			{
				// Now save to a new batch file and then re-read and re-open, to see (visualise) the difference
				std::string out_path;
				image_to_cifar(USER_DIR + original_image, out_path);
				
				// Reopen these "cifar10" saved files and redisplay
				std::vector<cv::Mat> batch2;
				cv::Mat label2 = cv::Mat::zeros(1, 10000, CV_64FC1);   
				read_batch(out_path, batch2, label2);
				
				std::string savedName2 = "batch_image1.png";
				std::cout << "Label2 = " << label2.at<int>(0, 0) << std::endl;
				
				imwrite(USER_DIR + savedName2, batch2[0]);
				
				imshow(savedName2, batch2[0]);
			}
			
			{
				// image_to_cifar2 does not work
				// And the same for image_to_cifar2
				/*std::string out_path;
				image_to_cifar2(USER_DIR + original_image, out_path); // Using second method
				
				std::vector<cv::Mat> batch3;
				cv::Mat label3 = cv::Mat::zeros(1, 10000, CV_64FC1);   
				read_batch(out_path, batch3, label3);
				
				std::string savedName3 = "batch_image2.png";
				std::cout << "Label3 = " << label3.at<int>(0, 0) << std::endl;
				
				imwrite(USER_DIR + savedName3, batch3[0]);
				
				imshow(savedName3, batch3[0]);*/
			}
			
			// Get the checksum of all the files
			std::string cmd;
			cmd = "sha1sum " + original_image;
			system(cmd.c_str());
			cmd = "sha1sum " + savedName2;
			system(cmd.c_str());
			//cmd = "sha1sum " + savedName3;
			//system(cmd.c_str());
			
			cv::waitKey(0);
		} 
		else if(mode_s == "create_batch") // Create a batch file (.bin file in cifar10 format) from a list of images
		{
			std::string out_path;
			std::vector<std::string> in_paths;
			std::vector<uint8_t> labels;
			
			in_paths.push_back(USER_DIR + "images/camera/truck/1.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/truck/2.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/truck/3.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/truck/4.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/truck/5.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/truck/6.jpeg");
			
			in_paths.push_back(USER_DIR + "images/camera/horse/1.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/horse/2.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/horse/3.jpeg");
			
			in_paths.push_back(USER_DIR + "images/camera/deer/1.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/deer/2.jpeg");	
			
			in_paths.push_back(USER_DIR + "images/camera/automobile/1.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/automobile/2.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/automobile/3.jpeg");
			in_paths.push_back(USER_DIR + "images/camera/automobile/4.jpeg");			
			
			labels = {9,9,9,9,9,9,7,7,7,4,4,1,1,1,1};
			
			images_to_cifar(in_paths, out_path, labels);
			
			std::cout << "Batch file saved to " << USER_DIR << out_path << std::endl;
		}
		else if(mode_s == "occlusion_test")
		{
			// Assign a default value and overwrite it if the user supplies the parameter
			std::string src_s = (src_arg ? args::get(src_arg) : "/home/xilinx/open_cv2/images/custom/cars/2.jpg");
			std::cout << "src: " << src_s << std::endl;
		
			unsigned int output;
			float certainty;
			
			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());

			// If first character is no a forward slash, then user is specifying a relative path
			// pre-concatinate the absolute path to the program
			if(src_s[0] != '/')
				src_s = "/home/xilinx/open_cv2/" + src_s;
			
			cv::Mat image = cv::imread(src_s);
			cv::imshow("Original Image", image);

			std::vector<std::string> classes = loadClasses();
			
			// Iteratively occlude and classify image
			int original_classification;
			std::cout << "Num columns = " << image.cols << std::endl;
			for (int i = 0; i < image.cols; i++)
			{
				// Save image to tmp file
				std::string tmp_path = USER_DIR + "tmp.png";
				cv::imwrite(tmp_path, image);
				
				std::string out_path;
				image_to_cifar(tmp_path, out_path); // Convert png to cifar10 format
				
				// Delete (clean up) the .png input file
				if(std::remove(tmp_path.c_str()) != 0)
					std::cout << "Failed to delete tmp file: " << tmp_path << std::endl;
				
				std::vector<unsigned int> detailed_results(classes.size());
				runBNN_image(out_path.c_str(), classes, output, certainty, detailed_results);
				double perc = ((double)i/image.cols) * 100;
				
				// original_classification is the classification of the image under no occlusion
				if(i == 0)
					original_classification = output;
		
				std::cout << "Classified as a " << classes[output] << std::endl;
				std::cout << "------------------" << std::endl;
				
				if(output != original_classification)
				{
					// Show the failed image
					std::cout << "Failed at perc = " << perc << std::endl;
					cv::imshow("Failed point", image);
					cvWaitKey(0);
				}
				
				// Delete .bin file
				if(std::remove(out_path.c_str()) != 0)
					std::cout << "Failed to delete tmp .bin file: " << out_path << std::endl;			

				//std::cout << "[" << classes[output] << "] - " << std::to_string(certainty) << " at occlusion = " << perc << "%" << std::endl;
				
				// Cover i rows of the image with all black pixels
				occludeImage(image, i);
			}
			
			// Just to check (should be all black)
			cv::imshow("Final occluded Image", image);
			cvWaitKey(0);
		}
		else if(mode_s == "noise_test")
		{
			// Assign a default value and overwrite it if the user supplies the parameter
			std::string src_s = (src_arg ? args::get(src_arg) : "/home/xilinx/open_cv2/images/custom/cars/2.jpg"); // 1.jpg performs poorly (as an example)
			std::cout << "src: " << src_s << std::endl;
		
			unsigned int output;
			float certainty;
			
			std::cout << "Initialising BNN with cifar10 weights" << std::endl;
			deinit();
			load_parameters(BNN_PARAM_DIR.c_str());

			// If first character is no a forward slash, then user is specifying a relative path
			// pre-concatinate the absolute path to the program
			if(src_s[0] != '/')
				src_s = "/home/xilinx/open_cv2/" + src_s;
			
			cv::Mat orig_image = cv::imread(src_s);
			cv::imshow("Original Image", orig_image);

			std::vector<std::string> classes = loadClasses();
			
			int original_classification;
			// Increase the standard deviation of the applied additive noise
			cv::Mat noisyImage = orig_image.clone();
			for (int i = 0; i < 100; i++)
			{
				// Save image to tmp file
				std::string tmp_path = USER_DIR + "tmp.png";
				cv::imwrite(tmp_path, noisyImage);
				
				std::string out_path;
				image_to_cifar(tmp_path, out_path); // Convert png to cifar10 format
				
				// Delete (clean up) the .png input file
				if(std::remove(tmp_path.c_str()) != 0)
					std::cout << "Failed to delete tmp file: " << tmp_path << std::endl;
				
				std::vector<unsigned int> detailed_results(classes.size());
				runBNN_image(out_path.c_str(), classes, output, certainty, detailed_results);
				
				if(i == 0)
					original_classification = output;
				
				std::cout << "std = " << i << std::endl;
				std::cout << "Classified as a " << classes[output] << std::endl;
				std::cout << "------------------" << std::endl;
				
				if(output != original_classification)
				{
					// Show the failed noisyImage
					std::cout << "Failed at std = " << i << std::endl;
					cv::imshow("Failed point", noisyImage);
					cvWaitKey(0);
				}	
				
				// Delete .bin file
				if(std::remove(out_path.c_str()) != 0)
					std::cout << "Failed to delete tmp .bin file: " << out_path << std::endl;			

				//std::cout << "[" << classes[output] << "] - " << std::to_string(certainty) << " at noise std = " << i << std::endl;
				
				addGaussianNoise(orig_image, noisyImage, 0, i);
			}
			
			// Just to check (should be all black)
			cv::imshow("Final noisy Image", noisyImage);
			cvWaitKey(0);
		}
		else
		{
			// Help message
			std::cout << parser;
		}
	}
	else
	{
		std::cout << parser;
	}
	
	return 0;
}

// TODO: Fix this
std::vector<std::string> loadClasses()
{
   	// Get a list of all the output classes
	std::vector<std::string> classes;
	std::ifstream file(BNN_PARAM_DIR + "/classes.txt");
	std::cout << "Opening parameters at: " << (BNN_PARAM_DIR + "/classes.txt") << std::endl;
	std::string str; 
	/*if (file.is_open())
  	{
		std::cout << "Classes: { ";
		while (std::getline(file, str))
		{
			std::cout << str << " ";
			classes.push_back(str);
		}
		std::cout << "}" << std::endl;
		file.close();
	}
	else
	{*/
		std::cout << "Unable to load classes. Resorting to known cifar10 " << std::endl;
		std::cout << "{ \"Airplane\", \"Automobile\", \"Bird\", \"Cat\", \"Deer\", \"Dog\", \"Frog\", \"Horse\", \"Ship\", \"Truck\" }" << std::endl;
		classes = { "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck" };
		//std::cout << "Failed to open classes.txt" << std::endl;
		//return -1;
	//}
	
	return classes;
}

// TODO: merge streamCamera and streamVideo
int streamVideo(roi_mode mode, std::vector<std::string> &classes, std::string src, bool save_output)
{
	cv::VideoCapture cap;
	if(src == "")
	{
		// If no src is supplied, open the camera
		cap.open(0); // Open the default camera
	}
	else
	{
		cap.open(src); // Open the video file
	}
	
	if(!cap.isOpened()) // Check if we succeeded
	{  
		std::cout << "ERROR! Unable to open the camera/video" << std::endl;
		return -1;
	}
	
	cv::VideoWriter outputVideo;
	if(save_output)
	{
		int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); // Get Codec Type- Int form
		cv::Size S = cv::Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),
							  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		outputVideo.open("output.avi", ex, cap.get(CV_CAP_PROP_FPS), S, true);
		
		if(!outputVideo.isOpened())
		{
			std::cout << "Could not open the output video for write" << std::endl;
			return -1;
		}
	}
		
	cv::Size frameSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	
	// Frames for each step
	cv::Mat curFrame, prevFrame, grayFrame, blurFrame;
	
	clock_t current_ticks, delta_ticks;
	clock_t fps   = 0;
	int frame_num = 0;
	
	std::vector<cv::Rect> prev_img_rois;
	
	while(true)
	{
		current_ticks = clock();
		
		cap >> curFrame;
		
		// If finished reading video file (not applicable when using camera)
		if (!curFrame.data)
			break;
		
 		// Resize image
		resize(curFrame, curFrame, frameSize); // Just overwrite the curFrame because no longer need the nonscaled version
		
		std::vector<cv::Rect> img_rois;
		// Each mode needs to take a frame and calculate the regions of interest as a vector of rectangles
		switch(mode)
		{
			case roi_mode::BACKGROUND_SUBTRACTION: {
				/*
				 *	Extract regions by diffusing and contouring the difference between adjacent frames
				 */
				 
				//cv::imshow("1. Original frame", curFrame);
				
				// Grayscale the image
				cv::cvtColor(curFrame, grayFrame, CV_BGR2GRAY);
				
				//cv::imshow("2. Grayscale frame", grayFrame);
				
				// Apply gaussian blue
				cv::blur(grayFrame, blurFrame, blurSize);
				
				//cv::imshow("3. Blurred frame", blurFrame);

				// Algorithm uses background subtraction, so needs a previous frame to run
				if (prevFrame.empty()) 
				{
					// Copy the current frame onto the previous frame
					blurFrame.copyTo(prevFrame);
					continue; // Move onto next frame
				}			
				
				// Note: prevFrame is a grayscale version of curFrame ( => grayFrame )
				cv::Mat diffFrame = backgroundSubtraction(blurFrame, prevFrame); // not using blurframe??

				// Extract the rectangles from the thresholded image
				img_rois = contourRegions(diffFrame);
				
				// Copy the current frame onto the previous frame
				blurFrame.copyTo(prevFrame); 
				
				//curFrame = diffFrame; // Temp
			break; }
		
			case roi_mode::STRIDE: {
				/*
				 * Tile the current frame and classify each tile. Classifications below the threshold will not be shown
				 */
				 
				// w, h, xstep, ystep
				img_rois = strideImage(curFrame, vp.box_width, vp.box_height, vp.stride_xstep, vp.stride_ystep);
			break; }
		
			case roi_mode::IMAGE_SIMPLIFICATION: {
				/*
				 * Simplify the colour scheme of the frame and then perform k-means clustering to identify regions
				 */
				 
				cv::Mat simpleFrame = imageSimplification(curFrame);
				
				cv::cvtColor(simpleFrame, grayFrame, CV_BGR2GRAY);
				
				img_rois = contourRegions(grayFrame);
				//curFrame = imageSimplification(curFrame); // Draw simplified frame
				//curFrame = simpleFrame; // Draw simplified frame
			break; }
				
			case roi_mode::CUSTOM_METHOD: {
				/*
				 * Experimental methods
				 */
				 
				cv::cvtColor(curFrame, grayFrame, CV_BGR2GRAY);
				
				if (!prevFrame.data)
				{
					grayFrame.copyTo(prevFrame);
					continue;
				}			
				
				// Note: prevFrame is a grayscale version of curFrame ( => grayFrame )
				cv::Mat diffFrame = backgroundSubtraction(grayFrame, prevFrame);

				std::vector<cv::Rect> img_rois_tmp = contourRegions(diffFrame);
				
				std::vector<cv::Rect> merged_img_rois;
				mergeOverlappingBoxes(img_rois_tmp, grayFrame, merged_img_rois);
				
				img_rois = {}; // Empty it
				for (auto const &m : merged_img_rois)
				{
					cv::Rect marker;
					// Make in shape of car
					marker.x = m.x;
					marker.y = m.y;
					marker.width = 200;
					marker.height = 100;
					
					// Need to check roi doesn't extend of the frame (will cause an error)
					if(marker.x < 0 || marker.y < 0 || marker.x + marker.width > WINDOW_WIDTH || marker.y + marker.height > WINDOW_HEIGHT)
						continue;
					
					img_rois.push_back(marker);
				}
				
				// Copy current frame to previous frame
				grayFrame.copyTo(prevFrame);
			break; }
				
			case roi_mode::SINGLE_CENTER_BOX: {
				/*
				 * There will be a centered box that will be classified for each frame
				 */
				cv::Rect center_box;
				//int w = 800;
				//int h = 400;
				
				// These width and heights work for the rotating car
				//int w = 600;
				//int h = 300;
				//int w = 300;
				//int h = 300;
				int w = vp.box_width;
				int h = vp.box_height;
				center_box.x = (WINDOW_WIDTH / 2) - (w / 2);
				center_box.y = (WINDOW_HEIGHT / 2) - (h / 2);
				center_box.width  = w;
				center_box.height = h;

				vp.bundle_regions = false; // Only one ROI, so don't bundle
				vp.threshold_certainty = 0; // Show whatever is classified, even if below threshold
				img_rois.push_back(center_box);
			break; }
				
		}
		
		if(roi_limit != -1)
			img_rois = getSubset(img_rois, roi_limit); // Only get the first roi_limit regions of interest
		
		// Classify the regions of interest
 		std::vector<unsigned int> outputs(img_rois.size()); // outputs are the classification of each region (highest index)
		std::vector<float> certainties(img_rois.size());

		std::vector<std::vector<unsigned int> > detailed_results;
		detailed_results.resize(img_rois.size());
		for (int i = 0; i < img_rois.size(); i++)
			detailed_results[i].resize(classes.size());
		
		auto t1 = std::chrono::high_resolution_clock::now();
		classifyRegions(curFrame, img_rois, classes, outputs, certainties, vp.bundle_regions, detailed_results);
		auto t2 = std::chrono::high_resolution_clock::now();
		
		bool use_tracking = true;
		if(use_tracking)
		{
			std::vector<int> flagged_objects(scene_objects.size(), 0); // Objects that have already been attended to this frame 
			for(int n = 0; n < img_rois.size(); n++)
			{
				bool new_object = true; // If this is a new region of interest
				int obj_ind; // The index of the scee object corresponding to this img_roi
				for(int j = 0; j < scene_objects.size(); j++)
				{
					// Check if this img_roi corresponds to this scene_object (if it overlaps)
					if((img_rois[n] & scene_objects[j].pos).area() > 0 && flagged_objects[j] == 0)
					{
						flagged_objects[j] = 1;
						obj_ind = j;
						new_object = false; // This img_roi corresponds to a pre-existing object
						
						// Add the classification result to the objects history
						scene_objects[j].result_history.insert(scene_objects[j].result_history.begin(), detailed_results[n]);
						
						if(scene_objects[j].result_history.size() > HST_LEN)
							scene_objects[j].result_history.pop_back(); // Remove any element older than HST_LEN frames
						
						// Similarly for prev_points
						scene_objects[j].prev_points.insert(scene_objects[j].prev_points.begin(), cv::Point(img_rois[n].x + img_rois[n].width/2, img_rois[n].y + img_rois[n].height/2));
						
						if(scene_objects[j].prev_points.size() > HST_LEN)
							scene_objects[j].prev_points.pop_back(); // Remove any element older than HST_LEN frames
					}
				}
				
				if(new_object)
				{
					// Add this img_roi as a new scene object
					scene_object so = scene_object();  
					so.pos = img_rois[n];
					so.prev_points.insert(so.prev_points.begin(), cv::Point(img_rois[n].x + img_rois[n].width/2, img_rois[n].y + img_rois[n].height/2));
					
					scene_objects.push_back(so);
					obj_ind = scene_objects.size() - 1;
					
					std::cout << "Found a new object in the scene" << std::endl;
				}
				
				// Update the output (classification) using weighted window
				std::cout << "------------------------------" << std::endl;
				for(int i = 0; i < scene_objects[obj_ind].result_history.size(); i++)
				{
					std::cout << "result_history[" << i << "]  = "; print_vector(scene_objects[obj_ind].result_history[i]); std::cout << std::endl;
				}
				
				// The current output is the maximum index of the weighted sum of current and previous outputs
				std::vector<unsigned int> adjusted_results(classes.size(), 0);
				for(int i = 0; i < scene_objects[obj_ind].result_history.size(); i++)
				{ // From most recent frame to oldest
					for(int j = 0; j < classes.size(); j++)
					{
						adjusted_results[j] += (history_weights[i] * scene_objects[obj_ind].result_history[i][j]);
					}
				}
				
				std::cout << "adjusted_results = "; print_vector(adjusted_results); std::cout << std::endl;
				
				unsigned int adjusted_output = getMaxIndex(adjusted_results);
				std::cout << "previously classified as a " << classes[outputs[n]] << std::endl;
				std::cout << "now classified as a = " << classes[adjusted_output] << std::endl;
				
				std::cout << "------------------------------" << std::endl;
				
				// Overwrite current classification with weighted sum
				outputs[n] = adjusted_output;
			}
		
			std::vector<unsigned int> empty_results(classes.size(), 0);
			for(int i = 0; i < flagged_objects.size(); i++)
			{
				if(flagged_objects[i] == 0)
				{
					// Will need to insert an empty set of classification results to shift them all back in time
					scene_objects[i].result_history.insert(scene_objects[i].result_history.begin(), empty_results);
					
					if(scene_objects[i].result_history.size() > HST_LEN)
						scene_objects[i].result_history.pop_back(); // Remove any element older than HST_LEN frames
				}
			}
			
		}
		
		//
		
		// Store previous classifications
		// THE FOLLOWING WORKS WELL WITH CAMERA!! DONT REMOVE
		/*if(mode == roi_mode::SINGLE_CENTER_BOX)
		{
			result_history.insert(result_history.begin(), detailed_results[0]);
			// Only one region of interest, so look at first index of detailed_results
			if(result_history.size() > HST_LEN)
				result_history.pop_back(); // Remove any element older than HST_LEN frames
			
			std::cout << "------------------------------" << std::endl;
			for(int i = 0; i < result_history.size(); i++)
			{
				std::cout << "result_history[" << i << "]  = "; print_vector(result_history[i]); std::cout << std::endl;
			}
			
			// The current output is the maximum index of the weighted sum of current and previous outputs
			std::vector<unsigned int> adjusted_results(classes.size(), 0);
			for(int i = 0; i < result_history.size(); i++)
			{ // From most recent frame to oldest
				for(int j = 0; j < classes.size(); j++)
				{
					adjusted_results[j] += (history_weights[i] * result_history[i][j]);
				}
			}
			
			std::cout << "adjusted_results = "; print_vector(adjusted_results); std::cout << std::endl;
			
			unsigned int adjusted_output = getMaxIndex(adjusted_results);
			std::cout << "previously classified as a " << classes[outputs[0]] << std::endl;
			std::cout << "now classified as a = " << classes[adjusted_output] << std::endl;
			
			std::cout << "------------------------------" << std::endl;
			
			// Overwrite
			outputs[0] = adjusted_output;
		}*/
		
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		if(vp.bundle_regions)
		{
			std::cout << "Pipelined frame classification took " << duration << " microseconds" << std::endl;
		}else{
			std::cout << "Not pipelined frame classification took " << duration << " microseconds" << std::endl;
		}

		cv::Scalar drawColour;
 		for(int i = 0; i < img_rois.size(); i++)
		{
			// If classification doesn't meet threshold, ignore it.
			if(certainties[i] < vp.threshold_certainty)
				continue;

			// Ignore some classifications
			//if(classes[outputs[i]] == "Ship" || classes[outputs[i]] == "Deer")
			//	continue;
			
			// Different classifications will have different rectangle colours
			drawColour = colourList[outputs[i]];
			
			// Draw the rectangle
			rectangle(curFrame, img_rois[i], drawColour, LINE_THICKNESS);
			//std::cout << "Rect: height=" << img_rois[i].height << ", width=" << img_rois[i].width << ", x=" << img_rois[i].x << ", y=" << img_rois[i].y << std::endl;
			
			// Label the rectangle with the result from the BNN
			std::string certainty_s = std::to_string(certainties[i]);
			//certainty_s.erase(certainty_s.find_last_not_of('0') + 2, std::string::npos); // Remove trailing zeros
			std::cout << classes[outputs[i]] + ": " + certainty_s << std::endl;
			int area = img_rois[i].width * img_rois[i].height;
			std::string text = classes[outputs[i]] + ": " + certainty_s; //+ " | " + std::to_string(area);
			putText(curFrame, text, cv::Point(img_rois[i].x, img_rois[i].y), cv::FONT_HERSHEY_PLAIN, 2, drawColour, LINE_THICKNESS);
		}	  
		
		// Draw all the scene objects as dots(RECTS) on the screen
		for(int i = 0; i < scene_objects.size(); i++)
		{
			int radius = 5;
			for (int j = 0; j < scene_objects[i].prev_points.size(); j++)
				circle(curFrame, cvPoint(scene_objects[i].prev_points[j].x, scene_objects[i].prev_points[j].y), radius, colourList[i % colourList.size()], -1, 8, 0);
		}
		
		// fps box at top right of screen
		/*std::stringstream fps_ss;
		fps_ss << "FPS: ";
		fps_ss << fps;
		putText(curFrame, fps_ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(255,255,0));
		std::cout << "FPS = " << fps << std::endl;*/
		
		std::stringstream num_objs_ss;
		num_objs_ss << "#Objects: ";
		num_objs_ss << scene_objects.size();
		putText(curFrame, num_objs_ss.str(), cv::Point(20, 50), cv::FONT_HERSHEY_PLAIN, 4, cv::Scalar(255,255,0), 2);

		/*std::stringstream frame_num_ss;
		frame_num_ss << "Frame Number: ";
		frame_num_ss << frame_num;
		putText(curFrame, frame_num_ss.str(), cv::Point(20, 80), cv::FONT_HERSHEY_PLAIN, 7, cv::Scalar(255,255,0)); 
		std::cout << "Frame Number = " << frame_num << std::endl;*/
		
		cv::imshow("Final video", curFrame);
		
		if(save_output)
		{
			// Write current frame to output file
			outputVideo << curFrame;
		}
		
		// Store previous extracted regions of interest
		prev_img_rois = img_rois;
		
		//if(frame_num == 43)
		//	cv::waitKey(0);
		
		frame_num++;
		
		delta_ticks = clock() - current_ticks; // Time in ms to read the frame, process it, and render it on the screen
		if (delta_ticks > 0)
			fps = CLOCKS_PER_SEC / delta_ticks;		
		
		int keyID = cvWaitKey(30); // need to change this 30
		//if (keyID != -1)
		//	keyPressed(keyID);
		//if(cv::waitKey(30) >= 0) break; // If no output on opencv screen, try increasing this
		
		//system("pause"); // Hold the frame until user presses a key in the terminal. *Not recognised on linux
	}
	
	// Finished, release the capture
	cap.release();
	cv::destroyAllWindows();
	
	return 0;
}
 
void keyPressed(int keyID)
{
	// Do something when a key is pressed ...
}

