#pragma once
#ifndef _OPCDR_MODELCONFIG_H_
#define _OPCDR_MODELCONFIG_H_

#include "MySqlBase.h"

struct ModelConfigAttr
{
	std::string id;
	std::string modelname;
	std::string aliasname;
};

class ModelConfig : public MySqlBase<ModelConfigAttr>
{
public:
	ModelConfig() = delete;
	ModelConfig(std::string hostName, std::string userName, std::string password);
	~ModelConfig();

public:
	virtual std::vector<ModelConfigAttr> getResult(sql::ResultSet* res);
	ModelConfigAttr getAttrByID(std::string id);
};

#endif