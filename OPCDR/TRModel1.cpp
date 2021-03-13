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
	//��ֱ�ӽ���ɸѡ����
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//��ͼ����й��ˣ�������ֵ�ĵ���ԭͼ��
	cv::threshold(mask, mask, threshold * maxVal, maxVal, cv::THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//��һ����0-255
	cv::Mat matForConn = mask.clone();
	cv::normalize(matForConn, matForConn, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//Ѱ����ͨ���lableͼ
	cv::Mat labels;
	//conn֪�������м�����ͨ������0������Ǳ�����1-(conn-1)������ǰ���Ĳ���
	int conn = cv::connectedComponents(matForConn, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//��ÿ����ͨ�����ֵ�����꣬���ж�����ֵ��ȡ��һ�����ֵ
	std::vector<float> maxValueConn(conn, 0);//����ÿ����ͨ������ֵ
	std::vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int* LinePtr = (int*)labels.ptr(i);
		float* LinePtrMask = (float*)mask.ptr(i);
		for (int j = 0; j < labels.cols; j++) {
			//�鿴�����������һ����ͨ��(1-(conn-1))
			int label = *(LinePtr + j);
			if (label == 0) {
				continue;
			}
			float value = *(LinePtrMask + j);
			//ֻ�д��ڵ�ʱ�򣬲Ż��¼�����ڵ�ʱ�򣬲����棬Ϊ�˱����Ժ�����ظ������ֵ��ֻȡ��һ�����ֵ
			if (value > maxValueConn[label]) {
				maxValueConn[label] = value;//�������ֵ
				points[label].x = j;//�������ֵ���±�
				points[label].y = i;
			}
		}
	}
	//���н�pointsתΪ512*512�еĵ�
	for (int i = 0; i < points.size(); i++) {
		points[i].x = int((points[i].x + 0.5) * (512 / mask_cols));
		points[i].y = int((points[i].y + 0.5) * (512 / mask_rows));
	}
	return points;//��ס����һ���㲻����ʲô����
}

void TRModel1::constructNetwork()
{
	//��һ���Ǻ�ӵ�һ�㣬Ϊ����Ӧ�Զ����resnet50
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