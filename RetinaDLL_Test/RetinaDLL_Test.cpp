#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "retina.h"

#pragma comment(lib, "opencv_world340.lib")
#pragma comment(lib, "RetinaDLL.lib")

int main()
{
	// my model, image path
	std::string model_path = "190728_retina_script_0.pt";
	std::string image_path = "img/1025.jpg";

	ObjectDetector _ObjectDetector;

	// my model load...
	_ObjectDetector.init(model_path);

	// read image
	cv::Mat img = cv::imread(image_path);
	cv::Mat result_img = img.clone();

	// image --> my script model --> receive pair<class_index, class_position>
	std::vector<std::pair<int, cv::Rect>> model_output = _ObjectDetector.objectDetect(img);

	for (int i = 0; i < model_output.size(); i++)
	{
		// print model output
		std::cout << "class index : " << model_output[i].first << ", " <<
			"x  y  width  height : " << 
			model_output[i].second.x << "  " <<
			model_output[i].second.y << "  " << 
			model_output[i].second.width << "  " <<
			model_output[i].second.height << std::endl;

		// draw
		cv::rectangle(result_img, model_output[i].second, cv::Scalar(0, 0, 255));
		cv::putText(result_img, std::to_string(model_output[i].first),
			cv::Point(model_output[i].second.x - 5, model_output[i].second.y - 2),
			0, 0.3,
			cv::Scalar(0, 0, 255));
	}
	cv::resize(result_img, result_img, cv::Size(result_img.cols * 4, result_img.rows * 4));
	cv::imshow("result", result_img);
	cv::waitKey();
	cv::destroyAllWindows();


	return 0;
}