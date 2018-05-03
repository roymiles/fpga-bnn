// Utility functions for OpenCV
// AUTHOR: ROY MILES (student)

#include "opencv_utils.h"
#include "common.h"
#include <thread>
#include <mutex>

#include "foldedmv-offload.h" // dilate-acc

using namespace cv;

/*
extern "C" {
	std::vector<uint8_t> doImageStuff_test(std::vector<uint8_t> &frames);
}
*/

video_parameters vp; 

std::vector<cv::Rect> contourRegions(cv::Mat &src)
{
	std::vector<std::vector<cv::Point>> cnts;
	
	findContours(src.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// The draw frame is dependant on the current viewing state
	//draw = *state2frame[curState];

	// Loop over the contours
	cv::Rect br;
	int output;
	float certainty;
	//cv::Mat bnnInput(bnnSize, CV_8UC1);

	int area;
	std::vector<cv::Rect> image_roi;
	//std::cout << "#Contours = " << cnts.size() << std::endl;
	for (auto &cnt : cnts)
	{
		// If the contour is too small, ignore it
		area = contourArea(cnt);

		// Don't waste computation
		if (area < vp.threshold_area || area > 200000)
			continue;

		// Compute the bounding box for the contour, draw it on the frame,
		// and update the text
		br = boundingRect(cnt);
			
		image_roi.push_back(br);
	}
	//std::cout << "#ROIS = " << image_roi.size() << std::endl;
	
	return image_roi;
}

const int NUM_REGIONS = (WINDOW_WIDTH/BLOCK_WIDTH) * (WINDOW_HEIGHT/BLOCK_HEIGHT);

// This concept was abandoned, however, someone could pick it up again
/*
void doImageStuff_thread(cv::Mat &result, cv::Mat &threshFrame, cv::Rect &roi) 
{
	std::vector<uint8_t> acc_input(BLOCK_HEIGHT * BLOCK_WIDTH);	
	std::vector<uint8_t> acc_output(BLOCK_HEIGHT * BLOCK_WIDTH);
	std::vector<cv::Mat> out_frames(NUM_REGIONS);
	
	cv::Mat m = threshFrame(roi); 

	mat2vector(m, acc_input);

	acc_output = doImageStuff_test(acc_input);
	
	cv::Mat img(BLOCK_HEIGHT, BLOCK_WIDTH, CV_8UC1);
	vector2mat<BLOCK_HEIGHT, BLOCK_WIDTH>(acc_output, img);
	
	cv::imshow("Test", img);

	// Update the output frame with the individual dilated regions
	// Loop through each pixel in the region of interest
	int x = 0;
	int y;
	for (int i = roi.x; i < roi.x + roi.width; i++)
	{
		y = 0;
		for (int j = roi.y; j < roi.y + roi.height; j++)
		{
			result.at<uint8_t>(j, i) = img.at<uint8_t>(y,x);
			y++;
		}
		x++;
	}	
}*/

cv::Mat backgroundSubtraction(cv::Mat &curImage, cv::Mat &prevImage)
{
	cv::Mat diffFrame = abs(curImage - prevImage);
	//cv::imshow("4. Difference frame", diffFrame);
	
	cv::Mat threshFrame;
	if(vp.adaptive_thresholding) {
		// Works well on highway because of the dark cars on dark background (so need high contrast to adapt)
		threshold(diffFrame, threshFrame, 0, 255, THRESH_BINARY | THRESH_OTSU); // Adaptive thresholding. Works well for highway, not street
	}else{
		std::cout << "threshold val = " << vp.threshold_val << std::endl;
		threshold(diffFrame, threshFrame, 70, 255, THRESH_BINARY); // Works well at threshold=70 for street
		std::cout << diffFrame.cols << "x" << diffFrame.rows << " " << threshFrame.cols << "x" << threshFrame.rows << std::endl;
	}
	//cv::imshow("5. Threshold frame", threshFrame);
	
	cv::Mat result(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC1);
	
	// The following block code is used if just the dilation is accelerated on the FPGA.
	// The input array will need to be partitioned appropriately before hand
/*
	auto t1 = chrono::high_resolution_clock::now();

	//const int NUM_THREADS = NUM_REGIONS;
	//std::thread t[NUM_THREADS];

	// Split the threshFrame into regions of size 120x180(width x height) from original 720x540
	// 6 in horizontal and 3 in vertical. 6 x 3 = 18 regions
	std::vector<cv::Rect> r;

	// Create the regions that are used to extract blocks from current and previous frame
	r.resize(NUM_REGIONS);
	int i = 0;
	for(int x = 0; x < WINDOW_WIDTH; x += BLOCK_WIDTH)
	{
		for(int y = 0; y < WINDOW_HEIGHT; y += BLOCK_HEIGHT)
		{
			r[i] = cv::Rect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT); // x,y,w,h
			i++;
		}
	}
	
	// Convert input frames to grayscale 8UC1
	//cv::Mat greyCur, greyPrev;
	//cv::cvtColor(curImage, greyCur, cv::COLOR_BGR2GRAY);
	//cv::cvtColor(prevImage, greyPrev, cv::COLOR_BGR2GRAY);
	
	// Pass each region into the accelerator
	//std::vector<uint8_t> acc_input(2 * BLOCK_HEIGHT * BLOCK_WIDTH);	// Accepts block of current and previous frame
	std::vector<uint8_t> acc_input(BLOCK_HEIGHT * BLOCK_WIDTH);
	//std::vector<uint8_t> acc_input_cur(BLOCK_HEIGHT * BLOCK_WIDTH);	
	//std::vector<uint8_t> acc_input_prev(BLOCK_HEIGHT * BLOCK_WIDTH);	
	std::vector<uint8_t> acc_output(BLOCK_HEIGHT * BLOCK_WIDTH);
	std::vector<cv::Mat> out_frames(NUM_REGIONS);
	//std::vector<cv::Mat> cur(NUM_REGIONS);
	//std::vector<cv::Mat> prev(NUM_REGIONS);
	std::vector<cv::Mat> m(NUM_REGIONS);
	cv::Mat result;
	
	//std::cout << "Starting threads" << std::endl;
	// Launch a group of threads NOT DOING THREADS ANYMORE
	for (int i = 0; i < NUM_REGIONS; ++i) {
		//t[i] = std::thread(doImageStuff_thread, i);
		doImageStuff_thread(result, threshFrame, r[i]);
	}
	
	//Join the threads with the main thread
	//for (int i = 0; i < NUM_THREADS; ++i) {
	//	t[i].join();
	//}
	//std::cout << "Joined last thread" << std::endl;
	
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
*/
	
	// Dilate the thresholded image to fill in holes, then find contours on thresholded image
	// This removes the noise (e.g. small bits of white that can be mistaken as ROI)
	//cv::Mat dilateFrame;
	//dilate(threshFrame, dilateFrame, cv::Mat(), cv::Point(-1, -1), _dilation_val);	
	
	//cv::Mat ex;
	// This morphological operation is equivelant to dilating and then eroding the image
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
	cv::morphologyEx(threshFrame, result, cv::MORPH_CLOSE, structuringElement);
	//auto t2 = chrono::high_resolution_clock::now();

	//auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
	//usecPerFrameDilation = (float)duration / (count);

	//std::cout << "Dilation time = " << (float)duration << std::endl;	
	
	//cv::imshow("6. Closed frame", result);
	
	return result; // ex | dilateFrame
}

int numXSteps = -1;
int numYSteps = -1;
std::vector<cv::Rect> strideImage(cv::Mat &img, int width, int height, int xStep, int yStep)
{

	std::cout << "strideImage { width = " << width << ", height = " << height << ", xStep = " << xStep << ", yStep = " << yStep << " }" << std::endl;
	std::cout << "strideImage { img.cols = " << img.cols << ", img.rows = " << img.rows << " }" << std::endl;

	if (numXSteps == -1)
		numXSteps = ceil((img.cols - width) / xStep);

	if (numYSteps == -1)
		numYSteps = ceil((img.rows - height) / yStep);

	int size = numXSteps * numYSteps;
	std::vector<cv::Rect> image_roi; // TODO: pre-calculate this size

	int i = 1;
	for (int x = 0; x < img.cols - width; x += xStep)
	{
		for (int y = 0; y < img.rows - height; y += yStep)
		{
			// Ignore top and bottom rows
			//if(y == 0 || y + yStep >= img.rows - height)
			//	continue;
			
			image_roi.push_back(cv::Rect(x, y, width, height));
			i++;
		}
	}

	return image_roi;
}

cv::Mat imageSimplification(cv::Mat &src)
{
	cv::Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<cv::Vec3b>(y, x)[z];


	int clusterCount = 10;
	cv::Mat labels;
	int attempts = 1;
	cv::Mat centers;

    auto t1 = chrono::high_resolution_clock::now();
	cv::kmeans(samples, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);
	auto t2 = chrono::high_resolution_clock::now();
	
    auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
	
	std::cout << "Took " << duration << "us to apply kmeans to the image " << std::endl;	
	
	cv::Mat new_image(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x*src.rows, 0);
			new_image.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	
	return new_image;
}

/*void image_to_cifar(std::string in, std::string &bufOut, uint8_t _label)
{
	bufOut = in + ".bin"; // Rename file extension
    std::ofstream output_file(bufOut);	
	
	// We resize the downloaded image to be 32x32 pixels as expected from the BNN
	cv::Mat img_in = cv::imread(in);
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
	std::vector<uint8_t> label = {_label};
	
	std::vector<uint8_t> out;
	out.insert(out.end(), label.begin(), label.end());
	out.insert(out.end(), r.rbegin(), r.rend());
	out.insert(out.end(), g.rbegin(), g.rend());
	out.insert(out.end(), b.rbegin(), b.rend());
	
    std::copy(out.rbegin(), out.rend(), std::ostream_iterator<char>(output_file));
}*/

void images_to_cifar(std::vector<std::string> in, std::string &bufOut, std::vector<uint8_t> labels)
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
		
		out.insert(out.end(), label.begin(), label.end());
		out.insert(out.end(), r.rbegin(), r.rend());
		out.insert(out.end(), g.rbegin(), g.rend());
		out.insert(out.end(), b.rbegin(), b.rend());
		
		i++;
	}
	
	std::copy(out.rbegin(), out.rend(), std::ostream_iterator<char>(output_file));
}

/*
 * DOES NOT WORK
 */
/*void image_to_cifar2(std::string in, std::string &bufOut)
{
	std::cout << "--------------- DO NOT USE THIS FUNCTION ---------------" << std::endl;
	cv::Mat img_in = cv::imread(in);
	cv::resize(img_in, img_in, cv::Size(32, 32), 0, 0, cv::INTER_CUBIC);
	cv::Mat img_out(cv::Size(32, 32), CV_8UC3);

	cv::Mat three_channels[3]; // Destination array
	cv::split(img_out, three_channels); // Split source 

	std::vector<uchar> output2(0);
	output2.push_back(1); // Class identifier is the first byte 0-9
	for (int colour = 2; colour >= 0; colour--) // 0=b, 1=g, 2=r. Need to store in rgb order -> 2,1,0
	{
		for (int i = 0; i < img_out.rows; i++)
		//for (int i = img_out.rows - 1; i >= 0; i--)
		{
			for (int j = 0; j < img_out.cols; j++)
			//for (int j = img_out.cols - 1; j >= 0; j--) 
			{
				uchar pix = three_channels[colour].at<uchar>(i, j);
				output2.push_back(pix);
			}
		}
	}

	bufOut = in + ".2.bin"; // Rename file extension
	std::ofstream output_file(bufOut);
	for (auto &px : output2)
		output_file << px;
	
	//system("ls");
	//cv::waitKey(0); // Wait until user enters a key on the console

	output_file.flush();
	output_file.close();
}*/

std::string saveImage(cv::Mat &img, bool toCifar)
{
	// Save image to tmp file
	std::string tmp_path = USER_DIR + "tmp" + GetUniqueId() + ".png";

//	std::cout << "Saving image at \"" << tmp_path << "\"" << std::endl;

	cv::imwrite(tmp_path, img);
	
	if(!toCifar)
		return tmp_path;
	
	std::string out_path;
	image_to_cifar(tmp_path, out_path); // Convert png to cifar10 format

	return out_path;
}

std::string saveImages(std::vector<cv::Mat> &imgs, bool toCifar)
{
	std::string tmp_path = USER_DIR + "tmp" + GetUniqueId() + ".png";
	
	return tmp_path;
}

void mergeOverlappingBoxes(std::vector<cv::Rect> &inputBoxes, cv::Mat &image, std::vector<cv::Rect> &outputBoxes)
{
	cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Mask of original image
	cv::Size scaleFactor(10, 10); // To expand rectangles, i.e. increase sensitivity to nearby rectangles. Doesn't have to be (10,10)--can be anything
	for (int i = 0; i < inputBoxes.size(); i++)
	{
		cv::Rect box = inputBoxes.at(i) + scaleFactor;
		cv::rectangle(mask, box, cv::Scalar(255), CV_FILLED); // Draw filled bounding boxes on mask
	}

	std::vector<std::vector<cv::Point>> contours;
	// Find contours in mask
	// If bounding boxes overlap, they will be joined by this function call
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int j = 0; j < contours.size(); j++)
	{
		outputBoxes.push_back(cv::boundingRect(contours.at(j)));
	}
}

// readCIFAR10.cc
// 
// feel free to use this code for ANY purpose
// author : Eric Yuan 
// my blog: http://eric-yuan.me/

// ----------------- Start ---
#define ATD at<double>

void 
read_batch(std::string filename, std::vector<cv::Mat> &vec, cv::Mat &label){
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int number_of_images = 10000;
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            std::vector<cv::Mat> channels;
            cv::Mat fin_img = cv::Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.ATD(0, i) = (double)tplabel;
        }
    }
}

cv::Mat 
concatenateMat(std::vector<cv::Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    cv::Mat res = cv::Mat::zeros(height * width, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        cv::Mat img(height, width, CV_64FC1);
        cv::Mat gray(height, width, CV_8UC1);
        cvtColor(vec[i], gray, CV_RGB2GRAY);
        gray.convertTo(img, CV_64FC1);
        // reshape(int cn, int rows=0), cn is num of channels.
        cv::Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        cv::Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    divide(res, 255.0, res);
    return res;
}

cv::Mat 
concatenateMatC(std::vector<cv::Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    cv::Mat res = cv::Mat::zeros(height * width * 3, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        cv::Mat img(height, width, CV_64FC3);
        vec[i].convertTo(img, CV_64FC3);
        std::vector<cv::Mat> chs;
        split(img, chs);
        for(int j = 0; j < 3; j++){
            cv::Mat ptmat = chs[j].reshape(0, height * width);
            Rect roi = cv::Rect(i, j * ptmat.rows, ptmat.cols, ptmat.rows);
            cv::Mat subView = res(roi);
            ptmat.copyTo(subView);
        }
    }
    divide(res, 255.0, res);
    return res;
}
// --------------- END ---

bool addGaussianNoise(const cv::Mat mSrc, cv::Mat &mDst, double Mean, double StdDev)
{
	if (mSrc.empty())
	{
		std::cout << "[Error]! Input Image Empty!" << std::endl;
		return 0;
	}
	cv::Mat mSrc_16SC;
	cv::Mat mGaussian_noise = cv::Mat(mSrc.size(), CV_16SC3);
	randn(mGaussian_noise, cv::Scalar::all(Mean), cv::Scalar::all(StdDev));

	mSrc.convertTo(mSrc_16SC, CV_16SC3);
	addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
	mSrc_16SC.convertTo(mDst, mSrc.type());

	return true;
}

void occludeImage(cv::Mat curImage, int numColumns)
{
	for (int i = 0; i < numColumns; i++)
	{
		// Set all  the rows in this column to black
		for (int j = 0; j < curImage.rows; j++)
		{
			curImage.at<Vec3b>(j, i)[0] = 0;
			curImage.at<Vec3b>(j, i)[1] = 0;
			curImage.at<Vec3b>(j, i)[2] = 0;
		}
	}
}