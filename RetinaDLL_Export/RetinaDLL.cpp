#include <opencv2/opencv.hpp>
#include "retina.h"
#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <string>
#include <iostream>
#include <memory>

# pragma comment(lib, "opencv_world340.lib")

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))


//			 === RetinaNet Object Detection === 
// My model load...
void ObjectDetector::init(std::string model_path)
{
	// jit model을 로딩.
	std::ifstream mymodel(model_path, std::ifstream::binary);
	ScriptModule = torch::jit::load(mymodel);

	assert(module != nullptr);
	std::cout << "\nSuccess Model Loading...\n";
}

// Model output <[predicted_index], [predicted_position]>
std::vector<std::pair<int, cv::Rect>> ObjectDetector::objectDetect(cv::Mat img)
{
	cv::Mat result_img = img.clone();
	cv::Mat my_img = img.clone();

	cv::cvtColor(my_img, my_img, cv::COLOR_BGR2RGB);

	cv::Mat img_float;
	my_img.convertTo(img_float, CV_32F, 1.0f / 255.0f);

	//// OpenCV Normalization
	//	for (int x = 0; x < img_float.cols; x++) {
	//		/*std::cout << 
	//			pointer_img_float[x + 0] << ", " <<
	//			pointer_img_float[x + 1] << ", " <<
	//			pointer_img_float[x + 2] << std::endl;*/
	//		pointer_img_float[x * 3 + 0] = (float)((float)(pointer_img_float[x * 3 + 0] - 0.485) / 0.229);
	//		pointer_img_float[x * 3 + 1] = (float)((float)(pointer_img_float[x * 3 + 1] - 0.456) / 0.224);
	//		pointer_img_float[x * 3 + 2] = (float)((float)(pointer_img_float[x * 3 + 2] - 0.406) / 0.225);
	//	}
	//}


	// Resize & Scaling
	std::cout << "\n=== Resize & Scaling === \n";
	int min_side = 608;
	int max_side = 1024;
	int rows = img_float.rows;
	int cols = img_float.cols;
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

	// Final Image Size
	int final_cols = (int)round(cols*scale);
	int final_rows = (int)round(rows*scale);
	std::cout << "final_cols : " << final_cols << std::endl;
	std::cout << "final_rows : " << final_rows << std::endl;
	cv::resize(img_float, img_float, cv::Size(final_cols, final_rows));

	// Padding (Resize rows, cols)
	std::cout << "\n=== Padding ====\n";
	int pad_w = 32 - (int)(img_float.rows % 32);
	int pad_h = 32 - (int)(img_float.cols % 32);

	std::cout << "pad_w : " << pad_w << std::endl;
	std::cout << "pad_h : " << pad_h << std::endl;

	// C++ OpenCV Zero Padding
	cv::copyMakeBorder(img_float, img_float,
		int(pad_w / 2),
		int(pad_w / 2),
		int(pad_h / 2),
		int(pad_h / 2),
		cv::BORDER_CONSTANT,
		cv::Scalar(0, 0, 0));

	// Mat --> Tensor
	torch::Tensor img_tensor = torch::from_blob(img_float.data,
		{ 1, img_float.rows, img_float.cols, 3 },
		torch::kFloat32).to(torch::kCUDA);

	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

	// Tensor Normalization
	img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);

	std::cout << "\n== Final Input Tensor Shape : " << img_tensor.sizes() << " ==" << std::endl;
	std::cout << "\n== Final Input Tensor Type : " << img_tensor.type() << " ==" << std::endl;

	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(img_tensor);

	// Module to gpu
	// Module to eval
	ScriptModule->to(at::kCUDA);
	ScriptModule->eval();

	// 3 Output Tensor
	c10::intrusive_ptr<c10::ivalue::Tuple> outputs = ScriptModule->forward(inputs).toTuple();
	torch::Tensor scores = outputs->elements()[0].toTensor().clone();
	torch::Tensor classification = outputs->elements()[1].toTensor().clone();
	torch::Tensor transformed_anchors = outputs->elements()[2].toTensor().clone();
	std::cout << "\n 3 Output received ... \n";

	// Number of Objects
	std::vector<int> idxs_vector;
	for (int64_t i = 0; i < scores.size(0); i++) {
		if (scores[i].item<float>() > 0.5) {
			idxs_vector.push_back(i);
		}
	}

	cv::Rect bbox;
	int class_idx;
	std::vector<std::pair<int, cv::Rect>> result_vector;

	// Number of Objects
	cv::resize(result_img, result_img, cv::Size(final_cols, final_rows));
	for (int i = 0; i < idxs_vector.size(); i++)
	{
		// Object Position
		int x1 = transformed_anchors[idxs_vector[i]][0].item<int>();
		int y1 = transformed_anchors[idxs_vector[i]][1].item<int>();
		int x2 = transformed_anchors[idxs_vector[i]][2].item<int>();
		int y2 = transformed_anchors[idxs_vector[i]][3].item<int>();

		// Predicted Index
		class_idx = classification[idxs_vector[i]].item<int>();

		// to original image size...
		bbox.x = int(x1 / scale);
		bbox.y = int(y1 / scale);
		bbox.width = int((x2 - x1) / scale);
		bbox.height = int((y2 - y1)  / scale);

		// push_back result
		result_vector.push_back(std::pair<int, cv::Rect>(class_idx, bbox));
	}
	std::cout << "\n Finished ... \n";

	return result_vector;

}