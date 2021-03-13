#include "TRModel1.h"

TRModel1::TRModel1(TRConfig trconfig):TRModel(trconfig)
{
	TRModel<M1DST>::build();
}

std::vector<cv::Point> getRegionPoints2(cv::Mat& mask, float threshold)
{
	int mask_cols = mask.cols;
	int mask_rows = mask.rows;
	//cout << "enter getRegionPoints2" <<endl;
	//先直接进行筛选操作
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//对图像进行过滤，大于阈值的等于原图像
	cv::threshold(mask, mask, threshold * maxVal, maxVal, cv::THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//归一化到0-255
	cv::Mat matForConn = mask.clone();
	cv::normalize(matForConn, matForConn, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//寻找连通域的lable图
	cv::Mat labels;
	//conn知道到底有几个连通域，其中0代表的是背景，1-(conn-1)，则是前景的部分
	int conn = cv::connectedComponents(matForConn, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//求每个连通域最大值的坐标，若有多个最大值，取第一个最大值
	std::vector<float> maxValueConn(conn, 0);//保存每个连通域的最大值
	std::vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int* LinePtr = (int*)labels.ptr(i);
		float* LinePtrMask = (float*)mask.ptr(i);
		for (int j = 0; j < labels.cols; j++) {
			//查看这个点属于哪一个连通域(1-(conn-1))
			int label = *(LinePtr + j);
			if (label == 0) {
				continue;
			}
			float value = *(LinePtrMask + j);
			//只有大于的时候，才会记录，等于的时候，不保存，为了避免以后会有重复的最大值，只取第一个最大值
			if (value > maxValueConn[label]) {
				maxValueConn[label] = value;//保留最大值
				points[label].x = j;//保留最大值的下标
				points[label].y = i;
			}
		}
	}
	//还有将points转为512*512中的点
	for (int i = 0; i < points.size(); i++) {
		points[i].x = int((points[i].x + 0.5) * (512 / mask_cols));
		points[i].y = int((points[i].y + 0.5) * (512 / mask_rows));
	}
	return points;//记住，第一个点不代表什么东西
}

void TRModel1::constructNetwork()
{
	//这一段是后加的一层，为了适应自定义的resnet50
	mParser->registerInput(mTrConfig.inputName.c_str(),
		nvinfer1::Dims3(mTrConfig.channel, mTrConfig.height, mTrConfig.width),
		nvuffparser::UffInputOrder::kNCHW);
	for (int i = 0; i < mTrConfig.outputName.size() - 1; i++)
	{
		mParser->registerOutput(mTrConfig.outputName[i].c_str());
	}
	mParser->parse(mTrConfig.modelPath.c_str(), *mNetwork);
	ITensor* outputTensor = mNetwork->getOutput(1);
	auto shuffle_layer = mNetwork->addShuffle(*outputTensor);
	Permutation permutation;
	for (int i = 0; i < Dims::MAX_DIMS; i++)
	{
		permutation.order[i] = 0;
	}
	permutation.order[0] = 2;
	permutation.order[1] = 0;
	permutation.order[2] = 1;
	shuffle_layer->setFirstTranspose(permutation);
	auto softmax_layer = mNetwork->addSoftMax(*shuffle_layer->getOutput(0));
	softmax_layer->getOutput(0)->setName(mTrConfig.outputName[2].c_str());
	//fileProp.outputNames.emplace_back("softmax/output");
	mNetwork->markOutput(*softmax_layer->getOutput(0));
}


std::vector<M1DST> TRModel1::OUT2DST(TROUT& out)
{
	std::vector<M1DST> ret;
	for (int i = 0; i < out.first; i++)
	{
		M1DST result;
		result.score = out.second[0][i];
		cv::Mat temp(16, 16, CV_32FC1, out.second[2].data() + 512 * i + 256);
		result.points = getRegionPoints2(temp, 0.7f);
		ret.emplace_back(result);
	}
	return ret;
}