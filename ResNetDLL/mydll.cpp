#include "build/mydll.hpp"
#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#include "opencv2/opencv.hpp"
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>

void Classifier::init()
{
	// script module load
	std::string model_path = "model/script_model_148.pt";
	std::ifstream mymodel(model_path, std::ifstream::binary);
	ScriptModule = torch::jit::load(mymodel);

	assert(module != nullptr);
	std::cout << "\nSuccess Model Loading...\n";
}

int Classifier::classify(cv::Mat img)
{
	// cv::Mat img : 3 channel BGR read
	// BGR -> RGB
	cv::Mat cvt_img;
	cv::cvtColor(img, cvt_img, cv::COLOR_BGR2RGB);

	// Normalization
	cv::Mat img_float;
	cvt_img.convertTo(img_float, CV_32F, 1.0f / 255.0f);

	// My ResNet Model Input Image Size
	cv::resize(img_float, img_float, cv::Size(32, 32));

	// Mat image to Tensor
	torch::Tensor img_tensor = torch::from_blob(
		img_float.data,
		{ 1, 32, 32, 3 },
		torch::ScalarType::Float).to(torch::kCUDA
		);

	// My Script Module Input : (1, 3, 32, 32) {CUDAFloatTensor}
	img_tensor = img_tensor.to(torch::kFloat);
	img_tensor = img_tensor.permute({ 0,3,1,2 });

	std::vector<torch::jit::IValue> inputs;
	// inputs.emplace_back(img_tensor);
	inputs.push_back(img_tensor);

	// std::cout << img_tensor << std::endl;
	std::cout << "\nSuccess Image Tensor load...\n";

	// GPU, Torch Evaluation
	ScriptModule->to(at::kCUDA);
	ScriptModule->eval();
	std::cout << "\nSuccess Scirpt Model to GPU...\n";

	torch::Tensor output = ScriptModule->forward(inputs).toTensor().clone().squeeze(0);
	std::cout << "\nSuccess Output Load...\n";

	torch::Tensor predict = torch::argmax(output);
	std::cout << "predict index : " << predict.item() << std::endl;

	// to int
	predicted_index = predict.item<int>();
	
	return predicted_index;
}
