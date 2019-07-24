#pragma once

#include "opencv2/opencv.hpp"
#include "torch/script.h"
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>
#include <string>
#include <iostream>
#include <memory>

#ifdef RESNETDLL_EXPORTS
	#define PYTORCH_API __declspec(dllexport)
#else
	#define PYTORCH_API __declspec(dllimport)
#endif

#pragma comment(lib, "opencv_world340.lib")

class Classifier
{
public:
	PYTORCH_API void init();
	PYTORCH_API int classify(cv::Mat img);
	// PYTORCH_API std::shared_ptr<torch::jit::script::Module> ModelLoad(std::string model_path);

private:
	// pytorch model
	std::shared_ptr<torch::jit::script::Module> ScriptModule;
	int predicted_index = 0;
};