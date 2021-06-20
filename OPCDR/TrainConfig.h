#pragma once
#ifndef _OPCDR_TRAINCONFIG_H_
#define _OPCDR_TRAINCONFIG_H_

#include "MySqlBase.h"

struct TrainConfigAttr
{
	std::string id;
	std::string aliasname;
	std::string sampleconfigid;
	std::string modelconfigid;
	std::string gpunumber;
	std::string epochnumber;
	std::string txtpath;
	std::string modelpath;
	std::string samplegroupid;
	std::string status;
	std::string uffpath;
	std::string dataperepoch;
	std::string pretrainedtrainconfigid;
	std::string pretrainedepochnumber;
	std::string dst_mpp;
	std::string dst_size;
	std::string spatial_augmentation;
	std::string stylish_augmentation;
	std::string normalization;
	std::string optimizer;
	std::string learning_rate;
	std::string lr_scheduler;
};

class TrainConfig : public MySqlBase<TrainConfigAttr>
{
public:
	TrainConfig() = delete;
	TrainConfig(std::string hostName, std::string userName, std::string password);
	~TrainConfig();

public:
	TrainConfigAttr getTrainConfigAttrByTrainId(std::string trainID);
	virtual std::vector<TrainConfigAttr> getResult(sql::ResultSet* res);
};

#endif