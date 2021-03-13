#pragma once
#ifndef _OPCDR_MODELRECOMMEND_H_
#define _OPCDR_MODELRECOMMEND_H_

#include "MySqlBase.h"

struct ModelRecommendAttr
{
    int id;
    std::string mid;
    int sid;
    std::string finished;
};

class ModelRecommend : public MySqlBase<ModelRecommendAttr>
{
public:
    ModelRecommend() = delete;
    ModelRecommend(std::string hostName, std::string userName, std::string password);
    ~ModelRecommend();
    bool insert(ModelRecommendAttr attr);
    bool updateStatus(ModelRecommendAttr attr);
    std::vector<ModelRecommendAttr> getModelUnFinishedSlide(std::string mid);

    virtual std::vector<ModelRecommendAttr> getResult(sql::ResultSet* res);
};

#endif