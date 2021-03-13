#include "TaskThread.h"
#include <windows.h>

std::vector<std::thread> TaskThread::pool = {};
std::map<std::string, TaskThread::TaskInfo> TaskThread::multi_tasks = {};
std::atomic<int> TaskThread::totalTaskNum = 0;
std::condition_variable TaskThread::task_cv = {};
std::mutex TaskThread::task_mutex = {};
std::atomic<bool> TaskThread::stopped = false;//停止线程的标志
std::atomic<int> TaskThread::idlThrNum = 8;//闲置线程数量
std::atomic<int> TaskThread::totalThrNum = 8;//总共线程数量
bool TaskThread::my_once_flag = false;//总共线程数量
int TaskThread::object_num = 0;

TaskThread::TaskThread() :key{MD5(getNanoseconds()).toStr() }
{
	object_num++;
	//std::string nano_time = getNanoseconds();
	//std::string md5_value = MD5(nano_time).toStr();
	//key = md5_value;
}

TaskThread::~TaskThread()
{
	while (multi_tasks.find(key) != multi_tasks.end()) {
		std::cout << "Task " << key << " left " << multi_tasks[key].tasks.size() << std::endl;
		Sleep(1000);
	}
	object_num--;
	if (object_num == 0) {
		std::cout << "deconstructing task thread" << std::endl;
		stopped.store(true);
		task_cv.notify_all();
		for (std::thread& thread : pool) {
			if (thread.joinable())
				thread.join();
		}
	}
	else {
		std::cout << "TaskThread has more child object, the father not released" << std::endl;
	}

}

void TaskThread::increaseRunningThread()
{
	std::unique_lock<std::mutex> lock{ this->task_mutex };
	if (multi_tasks.find(key) != multi_tasks.end())
	{
		multi_tasks[key].runningThreadNum++;
	}
}

void TaskThread::decreaseRunningThread(std::string current_key)
{
	std::unique_lock<std::mutex> lock{ this->task_mutex };
	//std::cout << "decrease running thread: " << key << std::endl;
	if (multi_tasks.find(current_key) != multi_tasks.end())
	{
		multi_tasks[current_key].runningThreadNum--;
		if (multi_tasks[current_key].runningThreadNum == 0)
		{
			//然后在判断tasks是否为0
			if (multi_tasks[current_key].tasks.size() == 0)
			{
				std::cout << "erase key " << current_key << std::endl;
				multi_tasks.erase(current_key);
			}
		}
	}
	else {
		return;
	}

}

void TaskThread::setThreadNum(int thread_num) {
	//std::unique_lock<std::mutex> locak{ this->task_mutex };
	multi_tasks[key].maxThreadNum = thread_num;
}

void TaskThread::createThreadPool(int thread_num)
{
	if (my_once_flag == true)
	{
		return;
	}
	my_once_flag = true;
	//在内部确定其只能被调用一次
	idlThrNum = thread_num;
	totalThrNum = thread_num;
	stopped.store(false);
	for (int size = 0; size < totalThrNum; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					std::string current_key;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->task_mutex };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->task_cv.wait(lock,
							[this] {
								//重大bug，如果totalTaskNum为零了，那么这里的就下不去，然后multi_tasks就不能被删除
								return this->stopped.load() || totalTaskNum != 0;
							}
						); // wait 直到有 task
						if (this->stopped.load() && this->multi_tasks.empty())
							return;
						//随机从map中取一个task
						auto it = multi_tasks.begin();
						srand((unsigned)time(NULL));
						std::advance(it, rand() % multi_tasks.size());
						auto random_key = it->first;

						//如果任务为空
						if (multi_tasks[random_key].tasks.empty()) {
							//如果没有正在执行的线程
							if (multi_tasks[random_key].runningThreadNum == 0) {
								//删除这个key
								multi_tasks.erase(random_key);
								std::cout << "erase multi tasks key: "<< random_key << std::endl;
								continue;
							}
							else
							{
								continue;
							}
						}
						//如果任务不为空
						else {
							//如果线程数量没有超过最大线程数量
							if (multi_tasks[random_key].runningThreadNum < multi_tasks[random_key].maxThreadNum) {
								task = std::move(multi_tasks[random_key].tasks.front());
								multi_tasks[random_key].tasks.pop();
								multi_tasks[random_key].runningThreadNum++;
								current_key = random_key;
							}
							else {
								continue;
							}
						}
						
					}
					idlThrNum--;
					//increaseRunningThread();
					task();
					totalTaskNum--;
					
					decreaseRunningThread(current_key);
					idlThrNum++;
				}
			}
			);
	}
}

void TaskThread::enterTask(std::queue<Task>& tasks, int thread_num)
{
	std::unique_lock<std::mutex> task_lock(TaskThread::task_mutex);
	totalTaskNum += tasks.size();
	while (!tasks.empty())
	{
		auto task = std::move(tasks.front());
		tasks.pop();
		multi_tasks[key].tasks.emplace(task);
	}
	setThreadNum(thread_num);
	task_lock.unlock();
	TaskThread::task_cv.notify_all();
}