#include "TFModel1.h"

TFModel1::TFModel1(TFConfig tfconfig) :TFModel(tfconfig) 
{

}

std::vector<cv::Point> TFModel1::getRegionPoints2(cv::Mat& mask, float threshold)
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
		points[i].x = int((points[i].x + 0.5) * (TFModel<M1DST>::mTfConfig.width / mask_cols));
		points[i].y = int((points[i].y + 0.5) * (TFModel<M1DST>::mTfConfig.height / mask_rows));
	}
	return points;//��ס����һ���㲻����ʲô����
}

void TFModel1::TensorToMat(tensorflow::Tensor mask, cv::Mat* dst)
{
	float* data = new float[(mask.dim_size(1)) * (mask.dim_size(2))];
	auto output_c = mask.tensor<float, 4>();
	//cout << "data 1 :" << endl;
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			data[j * mask.dim_size(1) + k] = output_c(0, j, k, 1);
		}
	}
	cv::Mat myMat = cv::Mat(mask.dim_size(1), mask.dim_size(2), CV_32FC1, data);
	*dst = myMat.clone();
	delete[]data;
}

std::vector<model1Result> TFModel1::resultOutput(std::vector<tensorflow::Tensor>& tensors)
{
	std::vector<model1Result> retResults;
	if (tensors.size() != 2)
	{
		std::cout << "model1Base::output: tensors size should be 2\n";
		return retResults;
	}
	auto scores = tensors[0].tensor<float, 2>();
	for (int i = 0; i < tensors[0].dim_size(0); i++)
	{
		model1Result result;
		cv::Mat dst2;
		TensorToMat(tensors[1].Slice(i, i + 1), &dst2);
		result.points = getRegionPoints2(dst2, 0.7);
		result.score = scores(i, 0);
		retResults.emplace_back(result);
	}
	return retResults;
}

std::vector<model1Result> TFModel1::OUT2DST(std::vector<tensorflow::Tensor>& out)
{
	std::vector<model1Result> tempResults = resultOutput(out);
	return tempResults;
}