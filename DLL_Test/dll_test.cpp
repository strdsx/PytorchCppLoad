#include "build/dll_test.h"
#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/highgui.h>

#pragma comment(lib, "opencv_world340.lib")
#pragma comment(lib, "ResNetDLL.lib")

#ifdef TEST
int main(int argc, const char* argv[])
{
	if (argc != 3) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}

	Classifier _Classifier;

    // My Script Module Load
	_Classifier.init();

    // Model Path
	std::string m_path = argv[1];

    // Image Path
	std::string img_path = argv[2];

	cv::Mat img = cv::imread(img_path);

	int predict_index = _Classifier.classify(img);

	std::cout << "predicted index : " << predict_index << std::endl;

	return 0;
}
