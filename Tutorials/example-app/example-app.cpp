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
// #define RESNET 1
// #define TEST 1
#define RETINA 1

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))




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

	img_tensor = img_tensor.to(torch::kFloat32);
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
	std::string model_path;
	std::string img_path;

	if (argc != 3) {
		model_path = "model/my_script.pt";
		img_path = "img/1025.jpg";
	}
	else {
		model_path = argv[1];
		img_path = argv[2];
	}

	// Script Module Load.
	std::ifstream mymodel(model_path, std::ifstream::binary);
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(mymodel);

	assert(module != nullptr);
	std::cout << "\nSuccess Model Loading..\n";

	// Model to GPU
	module->to(at::kCUDA);
	module->eval();

	cv::Mat img = cv::imread(img_path, 1);
	// std::cout << "== Origin image size : " << img.size() << " ==" << std::endl;
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	cv::Mat img_float;
	img.convertTo(img_float, CV_32F, 1.0f / 255.0f);

	/////////////////////////////////////
	// 여기서 normalization ??
	/////////////////////////////////////

	// Resize & Scaling
	std::cout << "\n=== Resize & Scaling === \n";
	int min_side = 608;
	int max_side = 1024;
	int rows = img_float.rows; //34
	int cols = img_float.cols; //132
	int cns = img_float.channels();

	int smallest_side = MIN(rows, cols);
	int largest_side = MAX(rows, cols);

	double scale = (double)min_side / (double)smallest_side;
	std::cout << "smallest_side : " << smallest_side << std::endl;
	printf("before scale : %.14f\n", scale);
	std::cout << "largest_side : " << largest_side << std::endl;

	if ((largest_side * scale) > max_side) {
		scale = (double)max_side / (double)largest_side;
	}
	printf("after scale : %.14f\n", scale);

	int result_cols = (int)round(cols*scale);
	int result_rows = (int)round(rows*scale);
	cv::resize(img_float, img_float, cv::Size(result_cols, result_rows));


	// Padding (Resize rows, cols)
	std::cout << "\n=== Padding ====\n";
	rows = img_float.rows;
	cols = img_float.cols;
	int pad_w = 32 - (int)(rows % 32);
	int pad_h = 32 - (int)(cols % 32);

	std::cout << "pad_w : " << pad_w << std::endl;
	std::cout << "pad_h : " << pad_h << std::endl;

	// C++ OpenCV Zero Padding
	cv::copyMakeBorder(img_float, img_float,
		int(pad_w / 2),
		int(pad_w / 2),
		int(pad_h / 2),
		int(pad_h / 2),
		cv::BORDER_CONSTANT,
		cv::Scalar(0,0,0));

	std::cout << "Padded Image cols, rows : " << img_float.cols << ", " << img_float.rows << std::endl;

	// Mat image --> (1, 34, 132, 3)
	torch::Tensor img_tensor = torch::from_blob(img_float.data,
		{1, img_float.rows, img_float.cols, 3},
		torch::kFloat32).to(torch::kCUDA);

	// --> (1, 3, 34, 132)
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

	// Normalization
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::cout << "\n== Final Input Tensor Shape : " << img_tensor.sizes() << " ==" << std::endl;
	std::cout << "\n== Final Input Tensor Type : " << img_tensor.type() << " ==" << std::endl;

	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);

	
	// 3 Output Tensor
	auto outputs = module->forward(inputs).toTuple();
	torch::Tensor scores = outputs->elements()[0].toTensor().clone();
	torch::Tensor classification = outputs->elements()[1].toTensor().clone();
	torch::Tensor transformed_anchors = outputs->elements()[2].toTensor().clone();

	std::cout << "\n=== Output === \n";
	std::cout << "scores : " << scores.sizes() << std::endl;
	std::cout << "classification : " << classification.sizes() << std::endl;
	std::cout << "transformed_anchors : "<< transformed_anchors.sizes() << std::endl;


	// idxs list (찾은 클래스 개수)
	std::vector<int> idxs_vector;
	for (int64_t i = 0; i < scores.size(0); i++) {
		if (scores[i].item<float>() > 0.5) {
			idxs_vector.push_back(i);
		}
	}

	cv::Rect bbox;
	cv::Mat result_img;
	cv::resize(img, result_img, cv::Size(result_cols, result_rows));
	for (int i = 0; i < idxs_vector.size(); i++)
	{
		int x1 = transformed_anchors[idxs_vector[i]][0].item<int>();
		int y1 = transformed_anchors[idxs_vector[i]][1].item<int>();
		int x2 = transformed_anchors[idxs_vector[i]][2].item<int>();
		int y2 = transformed_anchors[idxs_vector[i]][3].item<int>();
		int class_idx = classification[idxs_vector[i]].item<int>();

		bbox.x = x1;
		bbox.y = y1;
		bbox.width = x2 - x1;
		bbox.height = y2 - y1;
		std::cout << "x, y, width, height => " <<
			bbox.x << ", " <<
			bbox.y << ", " <<
			bbox.width << ", " <<
			bbox.height << std::endl;

		// Test
		cv::rectangle(result_img, bbox, cv::Scalar(0, 0, 255));
		cv::putText(result_img, std::to_string(class_idx), cv::Point(bbox.x - 5, bbox.y - 5), 0, 1, cv::Scalar(0, 0, 255));
	}

	cv::imshow("result image", result_img);
	cv::waitKey();
	cv::destroyAllWindows();


	// Argmax
	/*torch::Tensor predict = torch::argmax(output);
	std::cout << "\npredict value : " << torch::max(output) << std::endl;
	std::cout << "predict index : " << predict << std::endl;*/
#elif TEST
	// jit model을 로딩.
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