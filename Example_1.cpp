#include <fstream>
#include <iostream>

#include "opencv2/opencv.hpp"

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

class Myclass {
	
private:
	std::vector<int> aa;	
	
	friend class cereal::access;

	//template <class Archive>
	//void serialize(Archive & ar)
	//{
	//	ar(aa);
	//}

	template <class Archive>
	void save(Archive & ar) const
	{
		ar(aa);
	}

	template <class Archive>
	void load(Archive & ar)
	{
		ar(aa);
	}

public:

	Myclass() {}

	Myclass(int unt) {
		for (size_t i = 0; i < unt; i++)
		{
			aa.push_back(i+1);
		}
	}
	void show() {
		for (size_t i = 0; i < aa.size(); i++)
		{
			printf("aa[%zu] = %d\n", i, aa[i]);
		}
	}
};

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


int main()
{
	
	// ******************************************************** //
	// 					Saving to disk
	// ******************************************************** //
	Myclass m(3);
	std::vector<std::string> ss({ "testing", "some", "stringLongOneHere" });
	std::vector<cv::Rect> rr(9);
	for (size_t i = 0; i < rr.size(); i++)
	{
		rr[i] = cv::Rect(i + 1, i + 2, i + 3, i + 4);
	}
	cv::Mat img = cv::imread("D:/Datasets/CUHK_Square/frames_test/Culture_Square_70551.png");
	cv::Mat imgOrig = cv::imread("C:/Users/Kyaw/Desktop/photo/INRIA_pos_00001.png");
	cv::Mat roi(imgOrig, cv::Rect(0, 0, 64, 64));
	std::vector<cv::Mat> imgs({ img, roi });
	
	{
		std::ofstream fid("out.cereal", std::ios::binary);
		cereal::BinaryOutputArchive ar(fid);
		ar(m, ss, rr, imgs);
	}

	printf("Writing done.\n");
	
	
	// ******************************************************** //
	// 					Loading from disk
	// ******************************************************** //
	
	Myclass m2;	
	std::vector<std::string> ss2;
	std::vector<cv::Rect> rr2;
	std::vector<cv::Mat> imgs2;
	
	{
		std::ifstream fid("out.cereal", std::ios::binary);
		cereal::BinaryInputArchive ar(fid);
		ar(m2, ss2, rr2, imgs2);
	}

	printf("Reading done\n");
	
	m2.show();
	for (size_t i = 0; i < ss2.size(); i++)
	{
		std::cout << ss2[i] << std::endl;
	}
	for (size_t i = 0; i < rr2.size(); i++)
	{
		std::cout << rr2[i] << std::endl;
	}
	cv::imshow("img2", imgs2[0]); cv::imshow("roi2", imgs2[1]); cv::waitKey(0);

}
