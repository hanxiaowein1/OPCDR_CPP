#pragma once
#ifndef _OPCDR_MODELLIST_H_
#define _OPCDR_MODELLIST_H_

#include "MySqlBase.h"

class ModelListAttr
{
public:
	std::string id;
	std::string modelname;
	std::string input;
	std::string output;
	std::string output_size;
	std::string task;
public:
	ModelListAttr() {}
	//ModelListAttr(const ModelListAttr&) = delete;
	//ModelListAttr() {};
};

class ModelList : public MySqlBase<ModelListAttr>
{
public:
	ModelList() = delete;
	ModelList(std::string hostName, std::string userName, std::string password);
	~ModelList();

public:
	ModelListAttr getAttrByModelName(std::string modelname);
	virtual std::vector<ModelListAttr> getResult(sql::ResultSet* res);
};

#endif