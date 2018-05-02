// AUTHOR : ROY MILES (student-written)

#pragma once

#include <opencv2/core/core.hpp> // cv::Mat
#include <vector>
#include <string>

#include "common.h" // roi_mode, image_to_cifar

void classifyRegions(cv::Mat &curFrame, std::vector<cv::Rect> &regions, std::vector<std::string> &classes, std::vector<int> &outputs, std::vector<float> &certainties, bool bundleRegions, std::vector<std::vector<unsigned int> > &detailed_results);
void testBNN(const char* batch_path, int &output, float &certainty, int num_images, int pso);
//void runBNN_image(cv::Mat &img, std::vector<std::string> &classes, int &output, float &certainty);

/*
 * Pass in a path to the cifar10 formatted image
 * The output will determine the most likely classification
 */
void runBNN_image(const char* path, std::vector<std::string> &classes, unsigned int &output, float &certainty, std::vector<unsigned int> &detailed_results);
/*
 * Pass in a vector of cv::Mat
 * Each matrix will be converted to cifar10 format and then merged into a single .bin batch file
 */
void runBNN_multipleImages(std::vector<cv::Mat> &imgs, std::vector<std::string> &classes, std::vector<unsigned int> &outputs, std::vector<float> &certainties, std::vector<std::vector<unsigned int> > &detailed_results);

std::vector<std::string> loadClasses();
int streamVideo(roi_mode mode, std::vector<std::string> &classes, std::string src, bool save_output);
//int streamVideo(std::string videoPath, roi_mode mode = BACKGROUND_SUBTRACTION);
void keyPressed(int keyID);