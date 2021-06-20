#include "ModelConfig.h"

ModelConfig::ModelConfig(std::string hostName, std::string userName, std::string password) : MySqlBase(hostName, userName, password)
{
}

ModelConfig::~ModelConfig()
{
}

ModelConfigAttr ModelConfig::getAttrByID(std::string id)
{
	sql::ResultSet* res;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("select * from modelconfig where id = ?");
	pstmt->setString(1, id);
	res = select(pstmt);
	auto results = getResult(res);
	delete res;
	delete pstmt;

	ModelConfigAttr ret;
	if (results.size() > 0) {
		ret = results[0];
	}
	return ret;
}

std::vector<ModelConfigAttr> ModelConfig::getResult(sql::ResultSet* res)
{
	std::vector<ModelConfigAttr> ret;
	if (res == nullptr){
		std::cout << "ModelConfig::getResult: ResultSet is null" << std::endl;
		return ret;
	}
	while (res->next()) {
		ModelConfigAttr attr;
		attr.id = res->getString("id");
		attr.modelname = res->getString("modelname");
		attr.aliasname = res->getString("aliasname");
		ret.emplace_back(attr);
	}
	return ret;
}