#include "MultiImageRead.h"

MultiImageRead::MultiImageRead(const char* slidePath)
{
	m_slidePath = std::string(slidePath);
	key = "MultiImageRead_" + key;
}

MultiImageRead::~MultiImageRead()
{

}

void MultiImageRead::createReadHandle(int num)
{
	//TaskThread::setThreadNum(num);
	m_threadnum = num;
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	for (int i = 0; i < num; i++)
	{
		sReads.emplace_back(sFactory->createSlideProduct(m_slidePath.c_str()));
	}
	std::vector<std::mutex> list(sReads.size());
	sRead_mutex.swap(list);
}

void MultiImageRead::task(int i, cv::Rect rect)
{
	//std::cout << "MultiImageRead::task" << std::endl;
	std::unique_lock<std::mutex> sRead_lock(sRead_mutex[i]);
	std::unique_ptr<SlideRead>& uptr = sReads[i];
	std::pair<cv::Rect, cv::Mat> rectMat;
	uptr->getTile(read_level, rect.x, rect.y, rect.width, rect.height, rectMat.second);

	rectMat.first = rect;
	if (gamma_flag.load())
	{
		gammaCorrection(rectMat.second, rectMat.second, 0.6f);
	}

	PopQueueData<MIRData>::pushData(rectMat);
	//std::cout << "end one task\n";
}

void MultiImageRead::read(std::vector<cv::Rect> rects)
{
	std::queue<TaskThread::Task> tasks;
	int max_thread_num = 2;
	for (int i = 0; i < rects.size(); i++)
	{
		cv::Rect rect = rects[i];
		auto task = std::make_shared<std::packaged_task<void()>>(
			std::bind(&MultiImageRead::task, this, i % m_threadnum, rect));
		tasks.emplace(
			[task]() {
				(*task)();
			}
		);
	}
	TaskThread::enterTask(tasks, m_threadnum);
}

void MultiImageRead::gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma)
{
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = cv::saturate_cast<uchar>(int(pow((float)(i / 255.0), fGamma) * 255.0f));
	}
	//dst = src.clone();
	const int channels = dst.channels();
	switch (channels) {
	case 1: {
		cv::MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			*it = lut[(*it)];
		break;
	}
	case 3: {
		for (int i = 0; i < dst.rows; i++) {
			uchar* linePtr = dst.ptr(i);
			for (int j = 0; j < dst.cols; j++) {
				*(linePtr + j * 3) = lut[*(linePtr + j * 3)];
				*(linePtr + j * 3 + 1) = lut[*(linePtr + j * 3 + 1)];
				*(linePtr + j * 3 + 2) = lut[*(linePtr + j * 3 + 2)];
			}
		}
		break;
	}
	}
}

void MultiImageRead::getSlideWidth(int& width)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideWidth(width);
	}
}

void MultiImageRead::getSlideHeight(int& height)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideHeight(height);
	}
}

void MultiImageRead::getSlideBoundX(int& boundX)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideBoundX(boundX);
	}
}

void MultiImageRead::getSlideBoundY(int& boundY)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideBoundY(boundY);
	}
}

void MultiImageRead::getSlideMpp(double& mpp)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getSlideMpp(mpp);
	}
}

void MultiImageRead::getLevelDimensions(int level, int& width, int& height)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getLevelDimensions(level, width, height);
	}
}

void MultiImageRead::getTile(int level, int x, int y, int width, int height, cv::Mat& img)
{
	if (sReads.size() > 0)
	{
		sReads[0]->getTile(level, x, y, width, height, img);
	}
}

int MultiImageRead::get_ratio()
{
	if (sReads.size() > 0)
	{
		sReads[0]->ini_ration();
		return sReads[0]->m_ratio;
	}
	return -1;
}
