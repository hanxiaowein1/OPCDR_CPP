#include "TrainConfig.h"
#include <memory>

TrainConfig::TrainConfig(std::string hostName, std::string userName, std::string password) : MySqlBase(hostName, userName, password)
{
}

TrainConfig::~TrainConfig()
{}

TrainConfigAttr TrainConfig::getTrainConfigAttrByTrainId(std::string trainID)
{
	sql::ResultSet* res;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("select * from trainconfig where id=?");
	//std::unique_ptr<sql::PreparedStatement> pstmt(con->prepareStatement("select * from trainconfig where id=?"));
	pstmt->setString(1, trainID);
	res = select(pstmt);
	std::vector<TrainConfigAttr> results = getResult(res);
	TrainConfigAttr ret;
	if (results.size() > 0) {
		ret = results[0];
	}
	delete res;
	delete pstmt;
	return ret;
}

std::vector<TrainConfigAttr> TrainConfig::getResult(sql::ResultSet* res)
{
	std::vector<TrainConfigAttr> ret;
	if (res == nullptr) {
		std::cout << "TrainConfig::getResult: ResultSet is null!" << std::endl;
		return ret;
	}
	while (res->next()) {
		TrainConfigAttr attr;
		attr.id = res->getString("id");
		attr.aliasname = res->getString("aliasname");
		attr.sampleconfigid = res->getString("sampleconfigid");
		attr.modelconfigid = res->getString("modelconfigid");
		attr.gpunumber = res->getString("gpunumber");
		attr.epochnumber = res->getString("epochnumber");
		attr.txtpath = res->getString("txtpath");
		attr.modelpath = res->getString("modelpath");
		attr.samplegroupid = res->getString("samplegroupid");
		attr.status = res->getString("status");
		attr.uffpath = res->getString("uffpath");
		attr.dataperepoch = res->getString("dataperepoch");
		attr.pretrainedtrainconfigid = res->getString("pretrainedtrainconfigid");
		attr.pretrainedepochnumber = res->getString("pretrainedepochnumber");
		attr.dst_mpp = res->getString("dst_mpp");
		attr.dst_size = res->getString("dst_size");
		attr.spatial_augmentation = res->getString("spatial_augmentation");
		attr.stylish_augmentation = res->getString("stylish_augmentation");
		attr.normalization = res->getString("normalization");
		attr.optimizer = res->getString("optimizer");
		attr.learning_rate = res->getString("learning_rate");
		attr.lr_scheduler = res->getString("lr_scheduler");
		attr.id = res->getString("id");
		ret.emplace_back(attr);
	}
	return ret;
}