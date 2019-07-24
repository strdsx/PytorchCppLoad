#include "torch/script.h"
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#include <iostream>
#include <memory>
#include <string>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



# pragma comment(lib, "opencv_world340.lib")

#define INPUTSIZE 32
#define RESNET 1
// #define TEST 1
// #define RETINA 1



int main(int argc, const char* argv[]) 
{
#ifdef RESNET
	if (argc != 3) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}

	std::string model_path = argv[1];
	std::ifstream mymodel(model_path, std::ifstream::binary);
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(mymodel);

	// Loading Model
	assert(module != nullptr);
	// std::cout << "\nSuccess Model Loading\n" << std::endl;

	// Model to GPU
	module->to(at::kCUDA);
	module->eval();

	std::string img_path = argv[2];

	cv::Mat img = cv::imread(img_path, 1);
	std::cout << "== Origin image size : " << img.size() << " ==" << std::endl;
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	// Normalization
	cv::Mat img_float;
	img.convertTo(img_float, CV_32F, 1.0/255.0);
	cv::resize(img_float, img_float, cv::Size(INPUTSIZE, INPUTSIZE));
	std::cout << "== Input image size : " << img_float.size() << " ==" << std::endl;

	// (1, 3, 32, 32)
	torch::Tensor img_tensor = torch::from_blob(img_float.data, { 1, INPUTSIZE, INPUTSIZE, 3},
		torch::ScalarType::Float).to(torch::kCUDA);

	img_tensor = img_tensor.to(torch::kFloat);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });	

	std::cout << "\nimg tensor loaded...\n";

	/*std::cout << "\n== Input Tensor Shape : " << cuda_tensor.sizes() << " ==" << std::endl;
	std::cout << "\n== Input Tensor Type : " << cuda_tensor.type() << " ==" << std::endl;*/

	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);
	// inputs.push_back(img_tensor);

	torch::Tensor output = module->forward(inputs).toTensor().clone().squeeze(0);
			  	  
	// std::cout << output << std::endl;
	/*std::cout << "\n== Output Tensor Shape : " << output.sizes() << " ==" << std::endl;
	std::cout << "\n== Output Tensor Type : " << output.type() << " ==" << std::endl;*/

	torch::Tensor predict = torch::argmax(output);
	std::cout << "\npredict value : " << torch::max(output) << std::endl;
	std::cout << "predict index : " << predict.item() << std::endl;

#elif RETINA
	if (argc != 3) {
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
}

	std::string model_path = argv[1];
	std::ifstream mymodel(model_path, std::ifstream::binary);
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(mymodel);

	// Loading Model
	assert(module != nullptr);
	// std::cout << "\nSuccess Model Loading\n" << std::endl;

	// Model to GPU
	module->to(at::kCUDA);
	module->eval();

	std::string img_path = argv[2];

	cv::Mat img = cv::imread(img_path, 1);
	std::cout << "== Origin image size : " << img.size() << " ==" << std::endl;
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	// Normalization
	cv::Mat img_float;
	img.convertTo(img_float, CV_32F, 1.0f/255.0f);
	// cv::resize(img_float, img_float, cv::Size(1056, 320));
	std::cout << "== Input image size : " << img_float.size() << " ==" << std::endl;

	// (1, 40, 134, 3)
	torch::Tensor img_tensor = torch::from_blob(img_float.data, {1, img_float.cols, img_float.rows, 3},
		torch::kFloat32).to(torch::kCUDA);

	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	img_tensor = img_tensor.permute({ 0, 2, 3, 1 });

	std::cout << "\n== Input Tensor Shape : " << img_tensor.sizes() << " ==" << std::endl;
	std::cout << "\n== Input Tensor Type : " << img_tensor.type() << " ==" << std::endl;
	
	std::cout << img_tensor << std::endl;

	std::cout << "\nimg tensor loaded...\n";

	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);
	// inputs.push_back(img_tensor);

	torch::Tensor output = module->forward(inputs).toTensor().clone().squeeze(0);
	std::cout << "output success ...\n";

	std::cout << output << std::endl;

	// std::cout << output << std::endl;
	/*std::cout << "\n== Output Tensor Shape : " << output.sizes() << " ==" << std::endl;
	std::cout << "\n== Output Tensor Type : " << output.type() << " ==" << std::endl;*/

	torch::Tensor predict = torch::argmax(output);
	std::cout << "\npredict value : " << torch::max(output) << std::endl;
	std::cout << "predict index : " << predict << std::endl;
#elif TEST
	// jit modelÀ» ·Îµù.
	std::string model_path = "model/script_model_148.pt";
	std::ifstream mymodel(model_path, std::ifstream::binary);
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(mymodel);

	assert(module != nullptr);
	std::cout << "\nSuccess Model Loading...\n";

	cv::Mat img = cv::imread("img/3.jpg", 1);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	cv::Mat img_float;
	img.convertTo(img_float, CV_32F, 1.0 / 255.0);
	cv::resize(img_float, img_float, cv::Size(32, 32));

	torch::Tensor img_tensor = torch::from_blob(img_float.data, { 1, INPUTSIZE, INPUTSIZE, 3 },
		torch::ScalarType::Float).to(torch::kCUDA);

	img_tensor = img_tensor.to(torch::kFloat32);
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

	std::cout << "\nimg tensor loaded...\n";

	/*std::cout << "\n== Input Tensor Shape : " << cuda_tensor.sizes() << " ==" << std::endl;
	std::cout << "\n== Input Tensor Type : " << cuda_tensor.type() << " ==" << std::endl;*/

	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);
	// inputs.push_back(img_tensor);

	module->to(torch::kCUDA);
	torch::Tensor output = module->forward(inputs).toTensor().clone().squeeze(0);

	// std::cout << output << std::endl;
	/*std::cout << "\n== Output Tensor Shape : " << output.sizes() << " ==" << std::endl;
	std::cout << "\n== Output Tensor Type : " << output.type() << " ==" << std::endl;*/

	torch::Tensor predict = torch::argmax(output);
	std::cout << "\npredict value : " << torch::max(output) << std::endl;
	std::cout << "predict index : " << predict.item() << std::endl;
	
#endif

	return 0;
}
