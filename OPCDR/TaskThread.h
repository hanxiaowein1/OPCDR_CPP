#pragma once
#ifndef _AILAB_TASKTHREAD_H_
#define _AILAB_TASKTHREAD_H_

#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include <functional>
#include "MD5.h"
#include "MyTime.h"


class TaskThread
{
public:
	//alias
	using Task = std::function<void()>;
	//���������
	struct TaskInfo 
	{
		//�������
		std::queue<Task> tasks;
		//�������������ִ�е�����
		std::atomic<int> runningThreadNum = 0;
		//����������Ŀ�ʹ���߳�����
		std::atomic<int> maxThreadNum = 1;
	};

	//thread pool
	static std::vector<std::thread> pool;
	// ��������У�keyΪstring��valueΪ������������
	static std::map<std::string, TaskInfo> multi_tasks;
	//����������е��������
	static std::atomic<int> totalTaskNum;

	//�������������
	static std::condition_variable task_cv;
	//�������
	static std::mutex task_mutex;
	//ֹͣ�̵߳ı�־
	static std::atomic<bool> stopped;
	//�����߳�����
	static std::atomic<int> idlThrNum;
	//�ܹ��߳�����
	static std::atomic<int> totalThrNum;
	//ȷ������ִֻ��һ��
	static bool my_once_flag;
	//TaskThread�Ķ������
	static int object_num;

	//������е�key
	std::string key;
public:
	//static void initialize();
	//�����̳߳�
	void createThreadPool(int threadNum);
	//Ϊ���������������ʹ���߳���
	void setThreadNum(int threadNum);
public:
	//�����������ִ�е��߳�����һ
	virtual void decreaseRunningThread(std::string current_key);
	//�����������ִ�е��߳�����һ
	void increaseRunningThread();
	//Ϊ�������������񣬲��������ִ���߳���
	void enterTask(std::queue<Task> &tasks, int thread_num = 2);
public:
	TaskThread();
	virtual ~TaskThread();
};

#endif