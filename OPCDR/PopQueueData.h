#pragma once

#ifndef _AILAB_POPQUEUEDATA_H_
#define _AILAB_POPQUEUEDATA_H_

#include "TaskThread.h"

/// <summary>
/// ���̰߳�ȫ����
/// </summary>
/// <typeparam name="T">���������ݸ�ʽ</typeparam>
template <typename T>
class PopQueueData : public TaskThread
{
public:
	//���ݶ���	
	std::queue<T> data_queue;
	//���ݶ��л�����
	std::mutex data_mutex;
	//���ݶ�����������
	std::condition_variable data_cv;
public:
	void popQueueWithoutLock(std::vector<T>& pop_datas);
	void popQueueWithoutLock(T& pop_data);
	//�������ݶ��������е�����
	bool popData(std::vector<T>& pop_datas);
	//�������ݶ��еĵ�������
	bool popData(T& pop_data);
	//����
	void pushData(std::vector<T>& push_datas);
	//����
	void pushData(T& push_data);
	virtual ~PopQueueData();
};


template <typename T>
void PopQueueData<T>::popQueueWithoutLock(T& pop_data)
{
	if (!data_queue.empty())
	{
		pop_data = std::move(data_queue.front());
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::popQueueWithoutLock(std::vector<T>& pop_datas)
{
	int size = data_queue.size();
	for (int i = 0; i < size; i++)
	{
		pop_datas.emplace_back(std::move(data_queue.front()));
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::pushData(std::vector<T>& push_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	for (auto iter = push_datas.begin(); iter != push_datas.end(); iter++)
	{
		data_queue.emplace(std::move(*iter));
	}
	data_lock.unlock();
	data_cv.notify_one();
}

template <typename T>
void PopQueueData<T>::pushData(T& push_data)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	data_queue.emplace(std::move(push_data));
	data_lock.unlock();
	data_cv.notify_one();
}

template<typename T>
inline PopQueueData<T>::~PopQueueData()
{
}

template <typename T>
bool PopQueueData<T>::popData(T& pop_data)
{
	//std::cout << "------------------popData--------------------------" << std::endl;
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_data);
		data_lock.unlock();
		//std::cout << "111" << std::endl;
		return true;
	}
	else
	{
		data_lock.unlock();
		//ȡ��tasks����������Ƿ�������
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (multi_tasks.find(key) != multi_tasks.end())
		{
			//std::cout << "popData key:" << key;
			//std::cout << " , running thread num" << multi_tasks[key].runningThreadNum;
			//std::cout << ", tasks num" << multi_tasks[key].tasks.size() << std::endl;
			if (multi_tasks[key].runningThreadNum == 0)
			{
				if (multi_tasks[key].tasks.empty())
					return false;
			}
			task_lock.unlock();
			//֤����task����ôdata_mutex�ٴ����ϣ���Ϊһ���������ݻ�����
			data_lock.lock();
			using namespace std;
			while (!data_cv.wait_for(data_lock, 1000ms, [this] {
				if (data_queue.size() > 0 || stopped.load() || multi_tasks.find(key) == multi_tasks.end()) {
					return true;
				}
				return false;
				})) {
			}
			//data_cv.wait_for(data_lock, 1000ms, [this] {
			//	if (data_queue.size() > 0 || stopped.load() || multi_tasks.find(key) == multi_tasks.end()) {
			//		return true;
			//	}
			//	return false;
			//	});
			if (stopped.load() || multi_tasks.find(key) == multi_tasks.end())
				return false;
			//data_cv.wait(data_lock, [this] {
			//	if (data_queue.size() > 0 || stopped.load()) {
			//		return true;
			//	}
			//	else {
			//		return false;
			//	}
			//	});

			//std::cout << "1.555555" << std::endl;
			popQueueWithoutLock(pop_data);
			//std::cout << "222" << std::endl;
			data_lock.unlock();
			return true;
		}
		//������findkey��ʱ������ո����꣬Ȼ����ǡ�ý����˶��У����ʱ���������һ���ж�
		data_lock.lock();
		if (data_queue.size() > 0)
		{
			popQueueWithoutLock(pop_data);
			//std::cout << "333" << std::endl;
			data_lock.unlock();
			return true;
		}
		//std::cout << "error" << std::endl;
		return false;
	}
}

template <typename T>
bool PopQueueData<T>::popData(std::vector<T>& pop_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_datas);
		data_lock.unlock();
		return true;
	}
	else
	{
		data_lock.unlock();
		//ȡ��tasks����������Ƿ�������
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (TaskThread::multi_tasks.find(key) != TaskThread::multi_tasks.end())
		{
			//std::cout << "popData key:" << key;
			//std::cout << " , running thread num" << TaskThread::multi_tasks[key].runningThreadNum;
			//std::cout << ", tasks num" << TaskThread::multi_tasks[key].tasks.size() << std::endl;
			if (TaskThread::multi_tasks[key].runningThreadNum == 0)
			{
				if (multi_tasks[key].tasks.empty())
					return false;
			}
			task_lock.unlock();
			//֤����task����ôdata_mutex�ٴ����ϣ���Ϊһ���������ݻ�����(�����߳�Pop��ʱ��������һ���̻߳ᱻ ���ѣ�����������ܾͲ�������Ȼ�����ֹͣ)
			data_lock.lock();
			using namespace std;
			while (!data_cv.wait_for(data_lock, 1000ms, [this] {
				if (data_queue.size() > 0|| TaskThread::stopped.load()|| TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end()) {
					return true;
				}
				return false;
				})) {
				continue;
			}
			//data_cv.wait_for(data_lock, 1000ms, [this] {
			//	if (data_queue.size() > 0) {
			//		std::cout << "data queue size bigger than 0" << std::endl;
			//		return true;
			//	}
			//	if (TaskThread::stopped.load()) {
			//		std::cout << "stopped load" << std::endl;
			//		return true;
			//	}
			//	if (TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end()) {
			//		std::cout << "multi tasks key not find" << std::endl;
			//		return true;
			//	}
			//	std::cout << "data cv still wait" << std::endl;
			//	return false;
			//	});
			//data_cv.wait(data_lock, [this] {
			//	if (data_queue.size() > 0 || stopped.load()) {
			//		return true;
			//	}
			//	return false;
			//	});
			if (stopped.load() || TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end())
				return false;
			popQueueWithoutLock(pop_datas);
			data_lock.unlock();
			return true;
		}
		//������findkey��ʱ������ո����꣬Ȼ����ǡ�ý����˶��У����ʱ���������һ���ж�
		data_lock.lock();
		if (data_queue.size() > 0)
		{
			popQueueWithoutLock(pop_datas);
			data_lock.unlock();
			return true;
		}
		return false;
	}
}


#endif