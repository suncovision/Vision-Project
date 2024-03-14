#pragma once
#include <iostream>
#include <memory>
#include <chrono>
#include "cxxopts.hpp"


struct infos
{
    std::vector<int> location;
    std::string cls;
    std::string conf;
};

// new_add
static int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names) {
    DIR* p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline(infile, line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


std::vector<cv::Rect> Demo(cv::Mat& img,
    const std::vector<std::vector<Detection>>& detections,
    const std::vector<std::string>& class_names,
    std::vector<std::string>& cls,
    std::vector<std::string>& conf,
    bool label = true)
{
    std::cout << "detections is empty: " << detections.empty() << std::endl;
    if (!detections.empty()) 
    {
        std::vector<cv::Rect> boxes;

        for (const auto& detection : detections[0]) 
        {
            std::cout << "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-" << std::endl;
            
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            std::cout << "box: " << box << std::endl;
            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 1);
            
            if (label) 
            {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                cls.push_back(class_names[class_idx]);
                conf.push_back(ss.str());

                std::cout << "class: " << s << std::endl;

               /* auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline = 0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                    cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    cv::Point(box.tl().x + s_size.width, box.tl().y),
                    cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                    font_face, font_scale, cv::Scalar(255, 255, 255), thickness);*/
            }
            boxes.push_back(box);
        }
        return boxes;
    }
    else
    {
        std::vector<cv::Rect> ret;
        ret.push_back(cv::Rect(0, 0, 0, 0));
        return ret;
    }
}


int detect(std::vector<cv::Mat> images,
           infos &info, 
           std::vector< std::vector<infos>>&result_infos, 
           float conf_thres,
           int thre_roi,
           std::vector<std::string> class_names,
           Detector detector, 
           bool view_img)
{

    auto start = std::chrono::high_resolution_clock::now();

    int ok = 0;
    int ng = 0;
    for (size_t i = 0; i < images.size(); i += 1) {

        std::vector<infos> result_info;
        for (size_t j = i; j < i + 1 && j < images.size(); j++)
        {
            cv::Mat img = images[j];
            if (img.empty()) {
                std::cerr << "Error loading the image!\n";
                return -1;
            }

           
            // set up threshold
            //float conf_thres = 0.6;
            float iou_thres = 0.3;

            std::cout << "----------------------------------------------------" << std::endl;

            cv::Mat thr, src;
            //src = img.clone();   // 给软件要改
            cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);   // 给软件要改

            cv::threshold(src, thr, thre_roi, 255, cv::THRESH_BINARY);

            cv::Mat close;
            //cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            //cv::morphologyEx(thr, close, cv::MORPH_ERODE, ker);
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
            cv::morphologyEx(thr, close, cv::MORPH_ERODE, ker);
            //cv::morphologyEx(close, close, cv::MORPH_ERODE, ker);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(close, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            std::cout << "contours size: " << contours.size() << std::endl;
            std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> a, std::vector <cv::Point> b)
                {return cv::contourArea(a) > cv::contourArea(b); });

            cv::drawContours(src, contours, 0, cv::Scalar(255));
            cv::Rect rect = cv::boundingRect(contours[0]);

            cv::Rect RectTemp = rect;
            int bord = 0;
            if (thre_roi == 20)
                bord = 30;
           
            RectTemp.x = RectTemp.x - bord > 0 ? RectTemp.x - bord : 0;
            RectTemp.y = RectTemp.y - bord > 0 ? RectTemp.y - bord : 0;
            RectTemp.width = RectTemp.x + RectTemp.width + 2 * bord < src.cols ? RectTemp.width + 2 * bord : src.cols - RectTemp.x - 2;
            RectTemp.height = RectTemp.y + RectTemp.height + 2 * bord < src.rows ? RectTemp.height + 2 * bord : src.rows - RectTemp.y - 2;

            //cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);   // 给软件要改

            cv::Mat roi = img(RectTemp);
            cv::Mat res = roi.clone();

            int height = roi.rows;
            int width = roi.cols;
            
            int count = 0;
            std::vector<cv::Rect> box;
            std::vector<std::string> cls;
            std::vector<std::string> conf;
            for (int i = 0; i < height; i += 640)
            {
                if ((height - i) > 640)
                {
                    for (int j = 0; j < width; j += 640)
                    {
                        std::cout << i << ", " << j << std::endl;
                        if ((width - j) < 640)
                        {
                            cv::Rect region(j, i, width - j, 640);
                            cv::Mat mask = roi(region);
                            cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            auto result = detector.Run(mask, conf_thres, iou_thres);
                            // visualize detections
                            std::vector<cv::Rect> boxes;
                            if (view_img) {
                                boxes = Demo(mask, result, class_names, cls, conf);
                            }
                            if (boxes.size() != 0)
                            {
                                if (boxes[0] == cv::Rect(0, 0, 0, 0))
                                {
                                    box.push_back(boxes[0]);
                                }
                                else
                                {
                                    for (int k = 0; k < boxes.size(); k++)
                                    {
                                        boxes[k].x = boxes[k].x + j;
                                        boxes[k].y = boxes[k].y + i;
                                        box.push_back(boxes[k]);
                                    }
                                }
                            }
        
                            //mask.copyTo(resultImage(region));
                            std::cout << count << " -> " << "(" << i << "," << j << "," << width - j << ","
                                << 640 << ")" << std::endl;
                            count += 1;
                        }
                        else
                        {
                            cv::Rect region(j, i, 640, 640);
                            cv::Mat mask = roi(region);
                            cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            auto result = detector.Run(mask, conf_thres, iou_thres);
                            // visualize detections
                            std::vector<cv::Rect> boxes;
                            if (view_img) {
                                boxes = Demo(mask, result, class_names, cls, conf);
                            }
                            std::cout << "boxes size: " << boxes.size() << std::endl;

                            if (boxes.size() != 0)
                            {
                                if (boxes[0] == cv::Rect(0, 0, 0, 0))
                                {
                                    box.push_back(boxes[0]);
                                }
                                else
                                {
                                    for (int k = 0; k < boxes.size(); k++)
                                    {
                                        boxes[k].x = boxes[k].x + j;
                                        boxes[k].y = boxes[k].y + i;
                                        box.push_back(boxes[k]);
                                    }
                                }
                            }
                            //mask.copyTo(resultImage(region));
                            std::cout << count << " -> " << "(" << i << "," << j << "," << 640 << ","
                                << 640 << ")" << std::endl;
                            count += 1;
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < width; j += 640)
                    {
                        std::cout << i << ", " << j << std::endl;
                        if ((width - j) < 640)
                        {
                            cv::Rect region(j, i, width - j, height - i);
                            cv::Mat mask = roi(region);
                            cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            auto result = detector.Run(mask, conf_thres, iou_thres);
                            // visualize detections
                            std::vector<cv::Rect> boxes;
                            if (view_img) {
                                boxes = Demo(mask, result, class_names, cls, conf);
                            }

                            if (boxes.size() != 0)
                            {
                                if (boxes[0] == cv::Rect(0, 0, 0, 0))
                                {
                                    box.push_back(boxes[0]);
                                }
                                else
                                {
                                    for (int k = 0; k < boxes.size(); k++)
                                    {
                                        boxes[k].x = boxes[k].x + j;
                                        boxes[k].y = boxes[k].y + i;
                                        box.push_back(boxes[k]);
                                    }
                                }
                            }
                            //mask.copyTo(resultImage(region));
                            std::cout << count << " -> " << "(" << i << "," << j << "," << width - j << ","
                                << height - i << ")" << std::endl;
                            count += 1;
                        }
                        else
                        {
                            cv::Rect region(j, i, 640, height - i);
                            cv::Mat mask = roi(region);
                            cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                             auto result = detector.Run(mask, conf_thres, iou_thres);
                            // visualize detections
                            std::vector<cv::Rect> boxes;
                            if (view_img) {
                                boxes = Demo(mask, result, class_names, cls, conf);
                            }

                            if (boxes.size() != 0)
                            {
                                if (boxes[0] == cv::Rect(0, 0, 0, 0))
                                {
                                    box.push_back(boxes[0]);
                                }
                                else
                                {
                                    for (int k = 0; k < boxes.size(); k++)
                                    {
                                        boxes[k].x = boxes[k].x + j;
                                        boxes[k].y = boxes[k].y + i;
                                        box.push_back(boxes[k]);
                                    }
                                }
                            }
                            //mask.copyTo(resultImage(region));
                            std::cout << count << " -> " << "(" << i << "," << j << "," << 640 << ","
                                << height - i << ")" << std::endl;
                            count += 1;
                        }
                    }
                }
            }

            std::vector<int> flag;
            for (int i = 0; i < box.size(); i++)
            {
                if (box[i] == cv::Rect(0, 0, 0, 0))
                {
                    flag.push_back(0);
                }
                else
                {
                    flag.push_back(1);
                }
            }

            bool allZero = true;
            for (int i = 0; i < flag.size(); i++)
            {
                if (flag[i] != 0)
                {
                    allZero = false;
                    break;
                }
            }

            std::vector<cv::Rect> box_list;

            if (allZero)
            {
                std::vector<int> location;

                location.push_back(0);
                location.push_back(0);
                location.push_back(0);
                location.push_back(0);

                info.location = location;
                info.cls = "None";
                info.conf = "None";

                result_info.push_back(info);
            }
            else
            {
                for (int k = 0; k < flag.size(); k++)
                {
                    if (flag[k] != 0)
                    {
                        box_list.push_back(box[k]);
                    }
                }
            }

            std::cout << "box_list.size: " << box_list.size() << std::endl;
            std::cout << "cls.size: " << cls.size() << std::endl;
            std::cout << "conf.size: " << conf.size() << std::endl;

            for (int i = 0; i < box_list.size(); i++)
            {
                std::vector<int> location;
                box_list[i].x = box_list[i].x + RectTemp.x;
                box_list[i].y = box_list[i].y + RectTemp.y;

                location.push_back(box_list[i].x);
                location.push_back(box_list[i].y);
                location.push_back(box_list[i].width);
                location.push_back(box_list[i].height);

                info.location = location;
                info.cls = cls[i];
                info.conf = conf[i];

                result_info.push_back(info);
            }

            std::string route = "C:\\Users\\Lenovo\\Desktop\\9\\";
            cv::String filename = route + "dst_" + std::to_string(i) + ".bmp";
            cv::imwrite(filename, img);

            std::cout << "----------------------------------------------------" << std::endl;
        }
        result_infos.push_back(result_info);

    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // It should be known that it takes longer time at first time
    std::cout << "Total time consumed : " << duration.count() / 1000.0 << " s" << std::endl;
}

int detect1(std::vector<cv::Mat> images, infos& info,
    std::vector< std::vector<infos>>& result_infos, float conf_thres, int thre_roi,
    std::vector<std::string> class_names, Detector detector, bool view_img)
{
    //==============================================
    //bool is_gpu = "gpu";

    //bool view_img = "true";

    //// set device type - CPU/GPU
    //// set device type - CPU/GPU
    //torch::DeviceType device_type;
    //if (torch::cuda::is_available() && is_gpu) {
    //    device_type = torch::kCUDA;
    //}
    //else {
    //    device_type = torch::kCPU;
    //}

    //// load class names from dataset for visualization
    //std::vector<std::string> class_names = LoadNames("./weights/coco.names");   // 源文件为  ../weights/coco.names
    //if (class_names.empty()) {
    //    return -1;
    //}

    //// load network
    //std::string weights = "./weights/best.torchscript.pt";
    //auto detector = Detector(weights, device_type);

    //=====================================================

    // batch predict
    //std::cout << "img size: " << file_names.size() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    int ok = 0;
    int ng = 0;
    for (size_t i = 0; i < images.size(); i += 1) {
        std::vector<infos> result_info;
        for (size_t j = i; j < i + 1 && j < images.size(); j++)
        {
            cv::Mat img = images[j];
            if (img.empty()) {
                std::cerr << "Error loading the image!\n";
                return -1;
            }
            
            float iou_thres = 0.5;

            cv::Mat thr, src;
            //src = img.clone();   // 给软件要改
            cv::cvtColor(img, src, cv::COLOR_BGR2GRAY);   // 给软件要改

            cv::threshold(src, thr, 20, 255, cv::THRESH_BINARY);

            cv::Mat close;
            //cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            //cv::morphologyEx(thr, close, cv::MORPH_ERODE, ker);
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
            cv::morphologyEx(thr, close, cv::MORPH_ERODE, ker);
            //cv::morphologyEx(close, close, cv::MORPH_ERODE, ker);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(close, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            //std::cout << "contours size: " << contours.size() << std::endl;
            std::sort(contours.begin(), contours.end(), [](std::vector<cv::Point> a, std::vector <cv::Point> b)
                {return cv::contourArea(a) > cv::contourArea(b); });

            cv::drawContours(src, contours, 0, cv::Scalar(0, 0, 255));
            cv::Rect rect = cv::boundingRect(contours[0]);

            //cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);  // 给软件要改

            cv::Mat roi = img(rect);
            cv::Mat res = roi.clone();

            int height = roi.rows;
            int width = roi.cols;

            int count = 0;
            std::vector<cv::Rect> box;
            std::vector<std::string> cls;
            std::vector<std::string> conf;
            for (int i = 0; i < height; i += 640)
            {
                if ((height - i) > 640)
                {
                    for (int j = 0; j < width; j += 640)
                    {
                        //std::cout << i << ", " << j << std::endl;
                        if ((width - j) < 640)
                        {
                            cv::Rect region(j, i, width - j, 640);
                            cv::Mat mask = roi(region);
                            //cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            detector.Run1(mask, conf_thres, iou_thres);
                            //// visualize detections
                            //std::vector<cv::Rect> boxes;
                            //if (view_img) {
                            //    boxes = Demo(mask, result, class_names, cls, conf);
                            //}
                            //if (boxes.size() != 0)
                            //{
                            //    if (boxes[0] == cv::Rect(0, 0, 0, 0))
                            //    {
                            //        box.push_back(boxes[0]);
                            //    }
                            //    else
                            //    {
                            //        for (int k = 0; k < boxes.size(); k++)
                            //        {
                            //            boxes[k].x = boxes[k].x + j;
                            //            boxes[k].y = boxes[k].y + i;
                            //            box.push_back(boxes[k]);
                            //        }
                            //    }
                            //}

                            //mask.copyTo(resultImage(region));
                            /*std::cout << count << " -> " << "(" << i << "," << j << "," << width - j << ","
                                << 640 << ")" << std::endl;
                            count += 1;*/
                        }
                        else
                        {
                            cv::Rect region(j, i, 640, 640);
                            cv::Mat mask = roi(region);
                            //cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            detector.Run1(mask, conf_thres, iou_thres);
                            //// visualize detections
                            //std::vector<cv::Rect> boxes;
                            //if (view_img) {
                            //    boxes = Demo(mask, result, class_names, cls, conf);
                            //}
                            //std::cout << "boxes size: " << boxes.size() << std::endl;

                            //if (boxes.size() != 0)
                            //{
                            //    if (boxes[0] == cv::Rect(0, 0, 0, 0))
                            //    {
                            //        box.push_back(boxes[0]);
                            //    }
                            //    else
                            //    {
                            //        for (int k = 0; k < boxes.size(); k++)
                            //        {
                            //            boxes[k].x = boxes[k].x + j;
                            //            boxes[k].y = boxes[k].y + i;
                            //            box.push_back(boxes[k]);
                            //        }
                            //    }
                            //}
                            //mask.copyTo(resultImage(region));
                            /*std::cout << count << " -> " << "(" << i << "," << j << "," << 640 << ","
                                << 640 << ")" << std::endl;
                            count += 1;*/
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < width; j += 640)
                    {
                        //std::cout << i << ", " << j << std::endl;
                        if ((width - j) < 640)
                        {
                            cv::Rect region(j, i, width - j, height - i);
                            cv::Mat mask = roi(region);
                            //cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            detector.Run1(mask, conf_thres, iou_thres);
                            //// visualize detections
                            //std::vector<cv::Rect> boxes;
                            //if (view_img) {
                            //    boxes = Demo(mask, result, class_names, cls, conf);
                            //}

                            //if (boxes.size() != 0)
                            //{
                            //    if (boxes[0] == cv::Rect(0, 0, 0, 0))
                            //    {
                            //        box.push_back(boxes[0]);
                            //    }
                            //    else
                            //    {
                            //        for (int k = 0; k < boxes.size(); k++)
                            //        {
                            //            boxes[k].x = boxes[k].x + j;
                            //            boxes[k].y = boxes[k].y + i;
                            //            box.push_back(boxes[k]);
                            //        }
                            //    }
                            //}
                            //mask.copyTo(resultImage(region));
                            /*std::cout << count << " -> " << "(" << i << "," << j << "," << width - j << ","
                                << height - i << ")" << std::endl;
                            count += 1;*/
                        }
                        else
                        {
                            cv::Rect region(j, i, 640, height - i);
                            cv::Mat mask = roi(region);
                            //cv::imwrite("C:\\Users\\Lenovo\\Desktop\\img2\\" + std::to_string(count) + ".png", mask);
                            // inference
                            detector.Run1(mask, conf_thres, iou_thres);
                            //// visualize detections
                            //std::vector<cv::Rect> boxes;
                            //if (view_img) {
                            //    boxes = Demo(mask, result, class_names, cls, conf);
                            //}

                            //if (boxes.size() != 0)
                            //{
                            //    if (boxes[0] == cv::Rect(0, 0, 0, 0))
                            //    {
                            //        box.push_back(boxes[0]);
                            //    }
                            //    else
                            //    {
                            //        for (int k = 0; k < boxes.size(); k++)
                            //        {
                            //            boxes[k].x = boxes[k].x + j;
                            //            boxes[k].y = boxes[k].y + i;
                            //            box.push_back(boxes[k]);
                            //        }
                            //    }
                            //}
                            //mask.copyTo(resultImage(region));
                            /*std::cout << count << " -> " << "(" << i << "," << j << "," << 640 << ","
                                << height - i << ")" << std::endl;
                            count += 1;*/
                        }
                    }
                }
            }

            //std::vector<int> flag;
            //for (int i = 0; i < box.size(); i++)
            //{
            //    if (box[i] == cv::Rect(0, 0, 0, 0))
            //    {
            //        flag.push_back(0);
            //    }
            //    else
            //    {
            //        flag.push_back(1);
            //    }
            //}

            //bool allZero = true;
            //for (int i = 0; i < flag.size(); i++)
            //{
            //    if (flag[i] != 0)
            //    {
            //        allZero = false;
            //        break;
            //    }
            //}

            //std::vector<cv::Rect> box_list;

            //if (allZero)
            //{
            //    std::vector<int> location;

            //    location.push_back(0);
            //    location.push_back(0);
            //    location.push_back(0);
            //    location.push_back(0);

            //    info.location = location;
            //    info.cls = "None";
            //    info.conf = "None";

            //    result_info.push_back(info);
            //}
            //else
            //{
            //    for (int k = 0; k < flag.size(); k++)
            //    {
            //        if (flag[k] != 0)
            //        {
            //            box_list.push_back(box[k]);
            //        }
            //    }
            //    /*for (int i = 0; i < box.size(); i++)
            //    {

            //        std::vector<int> location;
            //        box[i].x = box[i].x + rect.x;
            //        box[i].y = box[i].y + rect.y;

            //        location.push_back(box[i].x);
            //        location.push_back(box[i].y);
            //        location.push_back(box[i].width);
            //        location.push_back(box[i].height);

            //        info.location = location;
            //        info.cls = cls[i];
            //        info.conf = conf[i];

            //        result_info.push_back(info);
            //    }*/
            //}

            //std::cout << "box_list.size: " << box_list.size() << std::endl;
            //std::cout << "cls.size: " << cls.size() << std::endl;
            //std::cout << "conf.size: " << conf.size() << std::endl;

            //for (int i = 0; i < box_list.size(); i++)
            //{
            //    std::vector<int> location;
            //    box_list[i].x = box_list[i].x + rect.x;
            //    box_list[i].y = box_list[i].y + rect.y;

            //    location.push_back(box_list[i].x);
            //    location.push_back(box_list[i].y);
            //    location.push_back(box_list[i].width);
            //    location.push_back(box_list[i].height);

            //    info.location = location;
            //    info.cls = cls[i];
            //    info.conf = conf[i];

            //    result_info.push_back(info);
            //}

            ////res(rect0).copyTo(resultImage(rect0));
            ////res(rect1).copyTo(resultImage(rect1));
            ////roi.copyTo(img(rect));
   /*         std::string route = "C:\\Users\\Lenovo\\Desktop\\9\\";
            cv::String filename = route + "dst" + std::to_string(i) + ".png";
            cv::imwrite(filename, img);*/

            //std::cout << "----------------------------------------------------" << std::endl;
        }
        result_infos.push_back(result_info);

    }
}
