#pragma once
#ifndef _OPCDR_SLIDEPREDICT_H_
#define _OPCDR_SLIDEPREDICT_H_

#include "Model.h"
#include "MultiImageRead.h"
#include "TaskThread.h"

struct SlideInfo
{
	int height   = 0;
	int width = 0;
	double mpp = 0.0f;
	double slideRatio = 0.0f;
};

//切片预测，传入一个模型，得到其所有的计算结果，其中保留着裁图的逻辑等
template <typename MLIN, typename MLOUT, typename SRC, typename DST>
//template <typename MODEL>
class SlidePredict : PopQueueData<std::pair<cv::Rect, cv::Mat>>
{
public:
	Model<MLIN, MLOUT, SRC, DST>* model;
	cv::Mat thumbnail;
public:
	SlidePredict(Model<MLIN, MLOUT, SRC, DST>* inModel);
	~SlidePredict();
	std::vector<std::pair<cv::Rect, DST>> run(MultiImageRead& mImgRead);
	int modelHeight;
	int modelWidth;
	float modelMpp;
private:
	bool initialize_binImg(MultiImageRead& mImgRead);
	void threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol);
	void remove_small_objects(cv::Mat& binImg, int thre_vol);
	bool iniPara(MultiImageRead& mImgRead);
	std::vector<cv::Rect> get_rects_slide();
	void pushData(MultiImageRead& mImgRead);
	std::vector<cv::Rect> iniRects(int sHeight, int sWidth, int height, int width, int overlap, bool flag_right, bool flag_down, int& rows, int& cols);
	void sort(std::vector<std::pair<cv::Rect, DST>>& results);
private:
	float model1OverlapRatio = 0.25f;
	int slideHeight;
	int slideWidth;
	double slideMpp;
	//裁取的宽高信息
	int block_height = 8192;
	int block_width = 8192;//在第0图层读取的图像的大小
	int read_level = 1;//model1读取的层级
	int levelBin = 4;
	double slideRatio;
	int m_crop_sum;
	cv::Mat binImg;
	int m_thre_col = 20;//rgb的阈值(与mpp无关)
	int m_thre_vol = 150;//面积的阈值(前景分割)
};

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>::SlidePredict(Model<MLIN, MLOUT, SRC, DST>* inModel)
{
	model = inModel;
	key = "SlidePredict_" + key;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>::~SlidePredict()
{
	delete model;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
std::vector<std::pair<cv::Rect, DST>> SlidePredict<MLIN, MLOUT, SRC, DST>::run(MultiImageRead& mImgRead)
{
	iniPara(mImgRead);
	initialize_binImg(mImgRead);
	mImgRead.setReadLevel(read_level);
	std::vector<cv::Rect> rects = get_rects_slide();
	if (rects.size() > 20)
	{
		rects.erase(rects.begin() + 20, rects.end());
	}
	mImgRead.read(rects);
	int temp_cols = 0;
	int temp_rows = 0;

	//使用陷阱，绝对不能先使用key，否则multi_tasks里面会遍历到这个key，然后发现没有任务就把他给删了。。。
	int max_thread_num = 2;
	std::queue<TaskThread::Task> tasks;
	for (int i = 0; i < max_thread_num; i++)
	{
		auto task = std::make_shared<std::packaged_task<void()>>
			(std::bind(&SlidePredict::pushData, this, std::ref(mImgRead)));
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
	}
	TaskThread::enterTask(tasks, max_thread_num);

	std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
	std::vector<std::pair<cv::Rect, DST>> ret;
	while (PopQueueData<std::pair<cv::Rect, cv::Mat>>::popData(rectMats))
	{
		std::vector<cv::Mat> input_imgs;
		std::vector<cv::Rect> input_rects;
		for (auto iter = rectMats.begin(); iter != rectMats.end(); iter++)
		{
			input_rects.emplace_back(std::move(iter->first));
			input_imgs.emplace_back(std::move(iter->second));
		}
		std::vector<DST> results;
		results = model->run(input_imgs);
		//std::cout << results.size() << std::endl;
		std::vector<std::pair<cv::Rect, DST>> final_results;
		for (auto iter = results.begin(); iter != results.end(); iter++)
		{
			std::pair<cv::Rect, DST> temp_result;
			int place = iter - results.begin();
			temp_result.first.x = input_rects[place].x * std::pow(slideRatio, read_level);
			temp_result.first.y = input_rects[place].y * std::pow(slideRatio, read_level);
			temp_result.first.width = modelWidth * (modelMpp / slideMpp);
			temp_result.first.height = modelHeight * (modelMpp / slideMpp);
			temp_result.second = *iter;
			final_results.emplace_back(temp_result);
		}
		ret.insert(ret.end(), final_results.begin(), final_results.end());

		rectMats.clear();
		//std::cout << ret.size() << std::endl;
	}
	sort(ret);
	return ret;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
bool SlidePredict<MLIN, MLOUT, SRC, DST>::iniPara(MultiImageRead& mImgRead)
{
	//初始化切片的宽高、mpp、ratio
	mImgRead.getSlideHeight(slideHeight);
	mImgRead.getSlideWidth(slideWidth);
	mImgRead.getSlideMpp(slideMpp);
	if (slideHeight <= 0 || slideWidth <= 0 || slideMpp <= 0)
		return false;
	//再来判断一些不合理的范围
	if (slideHeight > 1000000 || slideWidth > 1000000 || slideMpp > 1000000)
		return false;
	slideRatio = mImgRead.get_ratio();

	//初始化读取model1的level
	read_level = (modelMpp / slideMpp) / slideRatio;

	//初始化从哪一个level读取binImg
	double mySetMpp = 3.77f;//最原始的读取level4的mpp
	double compLevel = mySetMpp / slideMpp;
	std::vector<double> mppList;
	while (compLevel > 0.1f) {
		mppList.emplace_back(compLevel);
		compLevel = compLevel / slideRatio;
	}
	//遍历mppList，寻找与1最近的值
	double closestValue = 1000.0f;
	for (int i = 0; i < mppList.size(); i++) {
		if (std::abs(mppList[i] - 1.0f) < closestValue) {
			closestValue = std::abs(mppList[i] - 1.0f);
			levelBin = i;
		}
	}

	//初始化前景分割的阈值
	double mySetMpp2 = 0.235747f;//最原始的前景分割的mpp
	int thre_vol = 150;//最原始的面积阈值，在mySetMpp2上
	m_thre_vol = thre_vol / (slideMpp / mySetMpp2);
	int crop_sum = 960;//最原始的从binImg抠图的求和阈值，在mySetMpp2下
	m_crop_sum = crop_sum / (slideMpp / mySetMpp2);
	m_crop_sum = m_crop_sum / std::pow(slideRatio, levelBin);
	return true;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
std::vector<cv::Rect> SlidePredict<MLIN, MLOUT, SRC, DST>::get_rects_slide()
{
	std::vector<cv::Rect> rects;
	int constant1 = std::pow(slideRatio, read_level);

	int read_level_height = slideHeight / constant1;
	int read_level_width = slideWidth / constant1;
	//mImgRead.getLevelDimensions(read_level, read_level_width, read_level_height);
	int crop_width = 8192 / constant1;
	int crop_height = 8192 / constant1;
	if (crop_width > read_level_height || crop_height > read_level_width)
	{
		return rects;
	}

	int sHeight = modelHeight * float(modelMpp / slideMpp);
	int sWidth = modelWidth * float(modelMpp / slideMpp);
	sHeight = sHeight / constant1;
	sWidth = sWidth / constant1;
	//这个overlap要自适应
	//int overlap = 560 / constant1;
	//计算新的overlap
	int overlap_s = sWidth * model1OverlapRatio;
	int n = (crop_width - overlap_s) / (sWidth - overlap_s);
	int overlap = crop_width - n * (sWidth - overlap_s);

	int x_num = (read_level_width - overlap) / (crop_width - overlap);
	int y_num = (read_level_height - overlap) / (crop_height - overlap);


	std::vector<int> xStart;
	std::vector<int> yStart;
	bool flag_right = true;
	bool flag_down = true;
	if ((x_num * (crop_width - overlap) + overlap) == read_level_width) {
		flag_right = false;
	}
	if ((y_num * (crop_height - overlap) + overlap) == read_level_height) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((crop_width - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((crop_height - overlap) * i);
	}
	int last_width = read_level_width - x_num * (crop_width - overlap);
	int last_height = read_level_height - y_num * (crop_height - overlap);
	if (flag_right) {
		if (last_width >= sWidth)
			xStart.emplace_back((crop_width - overlap) * x_num);
		else {
			xStart.emplace_back(read_level_width - sWidth);
			last_width = sWidth;
		}
	}
	if (flag_down) {
		if (last_height >= sHeight)
			yStart.emplace_back((crop_height - overlap) * y_num);
		else {
			yStart.emplace_back(read_level_height - sHeight);
			last_height = sHeight;
		}
	}
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = crop_width;
			rect.height = crop_height;
			if (i == yStart.size() - 1) {
				rect.height = last_height;
			}
			if (j == xStart.size() - 1) {
				rect.width = last_width;
			}
			rects.emplace_back(rect);
		}
	}
	return rects;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void SlidePredict<MLIN, MLOUT, SRC, DST>::pushData(MultiImageRead& mImgRead)
{
	//std::cout << "enter slidepredict pushdata" << std::endl;
	std::vector<std::pair<cv::Rect, cv::Mat>> tempRectMats;
	int cropSize = modelHeight * float(modelMpp / slideMpp);
	cropSize = cropSize / std::pow(slideRatio, levelBin);
	while (mImgRead.popData(tempRectMats)) {
		std::vector<std::pair<cv::Rect, cv::Mat>> rectMats;
		for (auto iter = tempRectMats.begin(); iter != tempRectMats.end(); iter++) {
			//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\" + to_string(iter->first.x) + "_" + to_string(iter->first.y) + ".tif", iter->second);
			bool flag_right = false;
			bool flag_down = false;
			if (iter->second.cols != block_width / std::pow(slideRatio, read_level))
				flag_right = true;
			if (iter->second.rows != block_height / std::pow(slideRatio, read_level))
				flag_down = true;
			int crop_width = int(modelHeight * float(modelMpp / slideMpp)) / std::pow(slideRatio, read_level);
			int crop_height = int(modelWidth * float(modelMpp / slideMpp)) / std::pow(slideRatio, read_level);
			int overlap = (int(modelHeight * float(modelMpp / slideMpp)) / 4) / std::pow(slideRatio, read_level);
			int temp_rows = 0;
			int temp_cols = 0;
			std::vector<cv::Rect> rects = iniRects(
				crop_width, crop_height,
				iter->second.rows, iter->second.cols, overlap, flag_right, flag_down,
				temp_rows, temp_cols);

			for (auto iter2 = rects.begin(); iter2 != rects.end(); iter2++) {
				std::pair<cv::Rect, cv::Mat> rectMat;
				cv::Rect rect;
				rect.x = iter->first.x + iter2->x;
				rect.y = iter->first.y + iter2->y;
				//这里过滤掉在binImg中和为0的图像
				int startX = rect.x / std::pow(slideRatio, levelBin - read_level);
				int startY = rect.y / std::pow(slideRatio, levelBin - read_level);
				cv::Rect rectCrop(startX, startY, cropSize, cropSize);
				if (startX + cropSize > binImg.cols || startY + cropSize > binImg.rows)
				{
					rect.width = modelWidth;
					rect.height = modelHeight;
					rectMat.first = rect;
					rectMat.second = iter->second(*iter2);
					rectMats.emplace_back(std::move(rectMat));
					continue;
				}
				cv::Mat cropMat = binImg(rectCrop);
				int cropSum = cv::sum(cropMat)[0];
				if (cropSum <= m_crop_sum * 255)
				{
					continue;
				}
				rect.width = modelWidth;
				rect.height = modelHeight;
				rectMat.first = rect;
				rectMat.second = iter->second(*iter2);

				//在此处将其转为model所需要的大小
				cv::resize(rectMat.second, rectMat.second, cv::Size(modelWidth, modelHeight));

				rectMats.emplace_back(std::move(rectMat));
			}
		}
		PopQueueData<std::pair<cv::Rect, cv::Mat>>::pushData(rectMats);
		tempRectMats.clear();
	}
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
std::vector<cv::Rect> SlidePredict<MLIN, MLOUT, SRC, DST>::iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down,
	int& rows, int& cols)
{
	std::vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0) {
		std::cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width) {
		std::cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	if (overlap >= sWidth || overlap >= height) {
		std::cout << "overlap should < sWidth or sHeight\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	std::vector<int> xStart;
	std::vector<int> yStart;
	if ((x_num * (sWidth - overlap) + overlap) == width) {
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if (flag_right)
		xStart.emplace_back(width - sWidth);
	if (flag_down)
		yStart.emplace_back(height - sHeight);
	cols = xStart.size();
	rows = yStart.size();
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
			rects.emplace_back(rect);
		}
	}
	return rects;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
bool SlidePredict<MLIN, MLOUT, SRC, DST>::initialize_binImg(MultiImageRead& mImgRead)
{
	int heightL4 = 0;
	int widthL4 = 0;
	mImgRead.getLevelDimensions(levelBin, widthL4, heightL4);
	if (widthL4 == 0 || heightL4 == 0) {
		std::cout << "get L4 image failed\n";
		return false;
	}
	//cv::Mat imgL4;
	mImgRead.getTile(levelBin, 0, 0, widthL4, heightL4, thumbnail);
	threshold_segmentation(thumbnail, binImg, levelBin, m_thre_col, m_thre_vol);
	return true;
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void SlidePredict<MLIN, MLOUT, SRC, DST>::threshold_segmentation(cv::Mat& img, cv::Mat& binImg, int level, int thre_col, int thre_vol)
{
	//对img进行遍历，每三个unsigned char类型，选择其中的最大最小值
	std::unique_ptr<unsigned char[]> pBinBuf(new unsigned char[img.cols * img.rows]);
	unsigned char* pStart = (unsigned char*)img.datastart;
	unsigned char* pEnd = (unsigned char*)img.dataend;
	for (unsigned char* start = pStart; start < pEnd; start = start + 3)
	{
		//选择rgb元素中的最大最小值
		unsigned char R = *start;
		unsigned char G = *(start + 1);
		unsigned char B = *(start + 2);
		unsigned char maxValue = R;
		unsigned char minValue = R;
		if (maxValue < G)
			maxValue = G;
		if (maxValue < B)
			maxValue = B;
		if (minValue > G)
			minValue = G;
		if (minValue > B)
			minValue = B;
		if (maxValue - minValue > thre_col) {
			pBinBuf[(start - pStart) / 3] = 255;
		}
		else {
			pBinBuf[(start - pStart) / 3] = 0;
		}
	}
	binImg = cv::Mat(img.rows, img.cols, CV_8UC1, pBinBuf.get(), cv::Mat::AUTO_STEP).clone();
	//cv::imwrite("D:\\TEST_OUTPUT\\rnnPredict\\binImg_f.tif", binImg);
	//对binImg二值图进行操作
	remove_small_objects(binImg, thre_vol / pow(slideRatio, level));
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void SlidePredict<MLIN, MLOUT, SRC, DST>::remove_small_objects(cv::Mat& binImg, int thre_vol)
{
	//去除img中小的区域
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(binImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	double threshold = thre_vol;//面积的阈值
	std::vector<std::vector<cv::Point>> finalContours;
	for (int i = 0; i < contours.size(); i++) {
		double area = cv::contourArea(contours[i]);
		if (area >= threshold) {
			finalContours.emplace_back(contours[i]);
		}
	}
	if (finalContours.size() > 0) {
		cv::Mat finalMat(binImg.rows, binImg.cols, CV_8UC1, cv::Scalar(0));
		cv::fillPoly(finalMat, finalContours, cv::Scalar(255));
		binImg = finalMat.clone();
	}
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void SlidePredict<MLIN, MLOUT, SRC, DST>::sort(std::vector<std::pair<cv::Rect, DST>>& dsts)
{
	auto lambda = [](std::pair<cv::Rect, DST> dst1, std::pair<cv::Rect, DST> dst2)->bool {
		if (dst1.second > dst2.second)
			return true;
		return false;
	};
	std::sort(dsts.begin(), dsts.end(), lambda);
}

#endif