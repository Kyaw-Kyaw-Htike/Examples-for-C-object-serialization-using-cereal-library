#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include "opencv2/opencv.hpp"
#include "dlib/matrix.h"
#include "dlib/opencv.h"
#include "dlib/image_transforms.h"
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

namespace cv {

	template<class Archive>
	void serialize(Archive & ar, cv::Rect & m)
	{
		ar(m.x, m.y, m.width, m.height);
	}

	template<class Archive>
	void serialize(Archive & ar, cv::Size & m)
	{
		ar(m.width, m.height);
	}

	template<class Archive>
	void save(Archive& ar, const cv::Mat& mat)
	{
		int rows = mat.rows;
		int cols = mat.cols;
		int type = mat.type();
		bool continuous = mat.isContinuous();

		ar(rows, cols, type, continuous);

		if (continuous) {
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			ar(mat_data);
		}
		else {
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++) {
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				ar(row_data);
			}
		}
	};

	template<class Archive>
	void load(Archive& ar, cv::Mat& mat)
	{
		int rows, cols, type;
		bool continuous;

		ar(rows, cols, type, continuous);

		if (continuous) {
			mat.create(rows, cols, type);
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			ar(mat_data);
		}
		else {
			mat.create(rows, cols, type);
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++) {
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				ar(row_data);
			}
		}
	};
}

class Gen_samples_object_detection {

private:
	std::vector<cv::Mat> imgs_;
	std::vector<size_t> indices_imgs_;
	std::vector<cv::Rect> rois_;
	size_t ndata_;
	size_t idx_next_sample_;
	std::vector<size_t> shuffled_indices_;

	cv::Size winsize_;
	cv::Size winsize_final_;

	void shuffle_indices() {
		std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), std::mt19937{ std::random_device{}() });
	}

	bool rnd_coin_flip() {
		return std::rand() % 2; // for rnd_horz_flip and rnd_vert_flip
	}

	friend class cereal::access;

	template <class Archive>
	void save(Archive & ar) const
	{
		ar(imgs_, indices_imgs_, rois_, ndata_, idx_next_sample_, shuffled_indices_, winsize_, winsize_final_);
	}

	template <class Archive>
	void load(Archive & ar)
	{
		ar(imgs_, indices_imgs_, rois_, ndata_, idx_next_sample_, shuffled_indices_, winsize_, winsize_final_);
		srand(time(NULL));
	}

public:

	void create(

		const std::vector<cv::Mat>& imgs,
		const std::vector<std::vector<cv::Rect>>& bboxes,
		cv::Size winsize,
		cv::Size winsize_final,
		double thresh_overlap_ratio = 0.80,
		size_t stride_x = 1,
		size_t stride_y = 1,
		double scaleratio = std::pow(2, 1 / 8.0),
		size_t max_nscales = 100) {		

		size_t nimgs = imgs.size();

		bool checkWithGroundTruth = true;
		if (bboxes.empty())	checkWithGroundTruth = false;

		for (size_t idx_img = 0; idx_img < nimgs; idx_img++)
		{
			printf("============================\n");
			printf("Preparing; [%zu/%zu] completed...\n", idx_img + 1, nimgs);
			cv::Mat img = imgs[idx_img];
			printf("img width x height = %d x %d\n", img.cols, img.rows);
			std::vector<cv::Rect> bb;
			size_t nbb;
			if (checkWithGroundTruth) {
				bb = bboxes[idx_img];
				nbb = bb.size();
				printf("# of ground truth bounding boxes = %zu\n", nbb);
			}
			size_t ndata_collected_cur_img = 0;
			size_t s = 0;
			while (true) {
				double scale = std::pow(scaleratio, s);
				cv::Mat imgCur;
				cv::resize(img, imgCur, cv::Size(), 1.0 / scale, 1.0 / scale, cv::INTER_LINEAR);
				int w_imgCur = imgCur.cols;
				int h_imgCur = imgCur.rows;
				if (w_imgCur < winsize.width || h_imgCur < winsize.height) break;

				for (size_t i = 0; i < h_imgCur; i+=stride_y) {
					for (size_t j = 0; j < w_imgCur; j+=stride_x) {
						int x1 = std::round(j*scale);
						int y1 = std::round(i*scale);
						int w = std::round(winsize.width*scale);
						int h = std::round(winsize.height*scale);
						int x2 = x1 + w - 1;
						int y2 = y1 + h - 1;
						if (x2 >= w_imgCur - 1 || y2 >= h_imgCur - 1) continue;
						cv::Rect roi(x1, y1, w, h);

						bool save_this_roi = true;

						if (checkWithGroundTruth) {
							save_this_roi = false;
							for (size_t k = 0; k < nbb; k++)
							{
								double intersect_area = (roi & bb[k]).area();
								double union_area = (roi | bb[k]).area();
								double iou = intersect_area / union_area;
								if (iou >= thresh_overlap_ratio) {
									save_this_roi = true;
									break;
								}
							}
						}

						if (save_this_roi) {
							rois_.push_back(roi);
							indices_imgs_.push_back(idx_img);
							ndata_collected_cur_img++;
						}

					} //end for j
				} // end for i
				s++;
				if (s >= max_nscales) break;

			} // end while loop for multi-scale

			printf("# data points collected in this image = %zu\n", ndata_collected_cur_img);

		} // end for idx_img

		winsize_ = winsize;
		winsize_final_ = winsize_final;
		imgs_ = imgs;
		ndata_ = rois_.size();
		shuffled_indices_.resize(ndata_);
		std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
		shuffle_indices();
		idx_next_sample_ = 0;
		srand(time(NULL)); // for rnd_coin_flip() for rnd_horz_flip and rnd_vert_flip
		printf("Preparation completed.\n");

		printf("Total # of data points collected = %zu\n", ndata_);
		printf("Average # of data points collected per image = %.2f\n", (static_cast<double>(ndata_) / nimgs));
	}

	cv::Mat get_single(bool rnd_horz_flip=false, bool rnd_vert_flip=false) {
		if (idx_next_sample_ >= ndata_) {
			shuffle_indices();
			idx_next_sample_ = 0;
		}
		size_t idx_sampled = shuffled_indices_[idx_next_sample_];
		cv::Mat roi_sampled = imgs_[indices_imgs_[idx_sampled]](rois_[idx_sampled]).clone();
		if (rnd_horz_flip) {
			if (rnd_coin_flip()) cv::flip(roi_sampled, roi_sampled, 1);
		}
		if (rnd_vert_flip) {
			if (rnd_coin_flip()) cv::flip(roi_sampled, roi_sampled, 0);
		}
		cv::resize(roi_sampled, roi_sampled, winsize_final_);
		idx_next_sample_++;
		return roi_sampled;
	}

	std::vector<dlib::matrix<dlib::rgb_pixel>> get_batch(size_t batch_size, bool rnd_horz_flip = false, bool rnd_vert_flip = false) {
		std::vector<dlib::matrix<dlib::rgb_pixel>> imgs(batch_size);
		for (size_t i = 0; i < batch_size; i++)
			dlib::assign_image(imgs[i], dlib::cv_image<dlib::bgr_pixel>(get_single(rnd_horz_flip, rnd_vert_flip)));
		return imgs;
	}

	void visualize_single() {
		if (idx_next_sample_ >= ndata_) {
			shuffle_indices();
			idx_next_sample_ = 0;
		}
		size_t idx_sampled = shuffled_indices_[idx_next_sample_];
		cv::Mat img_sampled = imgs_[indices_imgs_[idx_sampled]];
		cv::Rect roi_sampled = rois_[idx_sampled];
		
		cv::rectangle(img_sampled, roi_sampled, cv::Scalar(255, 0, 0), 1);
		cv::imshow("visualize random sample", img_sampled);
		cv::waitKey(0);
		
		idx_next_sample_++;
	}

	void visualize_batch(size_t batch_size) {
		for (size_t i = 0; i < batch_size; i++)
			visualize_single();
	}
};


// **************************************************************** //
// 							main.cpp
// **************************************************************** //

#include <cstdio>
#include <iostream>
#include <string>
#include "dlib/gui_widgets.h"
#include "opencv2/opencv.hpp"

#include "data_io_object_detector_training_kkh.h"

#include <cereal/archives/binary.hpp>

using namespace std;
using namespace dlib;


int main() {
	
	// ***********************************************
	// 			Serialization
	// ***********************************************

	std::string fpath_xml = "data_annotation.xml";
		
	auto[imgs, bboxes, labels] = load_object_detection_image_dataset_dlib(fpath_xml, true, 0.9, true, true, 1.3, 1.3, 10, 10, true, 1);

	Gen_samples_object_detection gen;
	gen.create(imgs, bboxes, cv::Size(72 / 4, 80 / 4), cv::Size(72, 80), 0.80, 1, 1, std::pow(2, (1 / 8.0)));
	
	{
		std::ofstream fid("pos_sample_generator.cereal", std::ios::binary);
		cereal::BinaryOutputArchive ar(fid);
		ar(gen);
	}
	
	// ***********************************************
	// 			DeSerialization
	// ***********************************************

	gen_samples_object_detection gen2;
	{
		std::ifstream fid("pos_sample_generator.cereal", std::ios::binary);
		cereal::binaryinputarchive ar(fid);
		ar(gen2);
	}

	kkh::stdlib::Timer tt;
	std::vector<dlib::matrix<dlib::rgb_pixel>> crops;

	crops = gen2.get_batch(16, true);

	dlib::image_window win;
	for (size_t i = 0; i < crops.size(); i++)
	{
		win.set_image(crops[i]);
		cin.get();
	}

}