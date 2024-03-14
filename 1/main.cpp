#include "detector.h"
#include "test.h"
#include <vector>
#include <fstream>


int main()
{
	system("color 3F");

	std::vector<std::string> class_names = LoadNames("./weights/coco.names");   // Դ�ļ�Ϊ  ../weights/coco.names
	if (class_names.empty()) {
		return -1;
	}
	bool view_img = 1;
	torch::DeviceType device_type = torch::kCUDA;
	//std::string weights = "./weights/best.torchscript.pt";
	std::string weights = "E:\\PyCharm 2023.1\\project\\yolov5-6.0_modify\\runs\\train\\exp81\\weights\\best.torchscript.pt";
	auto detector = Detector(weights, device_type);
	
	cv::String path0 = "./images";
	std::vector<cv::String> files0;
	cv::glob(path0, files0, false);
	std::vector<cv::Mat> images0;
	for (int i = 0; i < files0.size(); i++)
	{
		cv::Mat src = cv::imread(files0[i]);
		images0.push_back(src);
	}
	
	infos info;
	std::vector<std::vector<infos>> result_infos;   //���������ÿ��ͼ����ÿ��ȱ�ݵļ����Ϣ
	float conf_thres = 0.50;  // ���Ŷ���ֵ����ֵԽ�󣬼����Խ��
	int thre_roi = 30;
	std::cout << "��ʼ����ʼ..." << std::endl;
	detect1(images0, info, result_infos, conf_thres, thre_roi, class_names, detector, view_img);
	std::cout << "��⿪ʼ" << std::endl;
	cv::String path = "C:\\Users\\Lenovo\\Desktop\\6";
	std::vector<cv::String> files;
	cv::glob(path, files, false);
	
	std::vector<cv::Mat> images;
	for (int i = 0; i < files.size(); i++)
	{
		cv::Mat src = cv::imread(files[i]);
		images.push_back(src);
	}
	std::cout << "image size: " << images.size() << std::endl;
	detect(images, info, result_infos, conf_thres, thre_roi, class_names, detector, view_img);
	
	for (int i = 0; i < result_infos.size(); i++)
	{
		std::cout << "\n===========================" << std::endl;
		for (int j = 0; j < result_infos[i].size(); j++)
		{
			std::cout << result_infos[i][j].location << std::endl;  // (���Ͻ�x�����Ͻ�y������)
			std::cout << result_infos[i][j].cls << std::endl;       // ���
			std::cout << result_infos[i][j].conf << std::endl;      // ���Ŷ�
		}
	}

	return 0;
}