#include "ModelList.h"

ModelList::ModelList(std::string hostName, std::string userName, std::string password) : MySqlBase(hostName, userName, password)
{
}

ModelList::~ModelList()
{}

ModelListAttr ModelList::getAttrByModelName(std::string modelname)
{
	using namespace std;
	sql::ResultSet* res;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("select * from modellist where modelname=?");
	pstmt->setString(1, modelname);
	res = select(pstmt);
	std::vector<ModelListAttr> results = getResult(res);
	ModelListAttr attr;
	if (results.size() > 0) {
		attr = results[0];
	}
	delete res;
	delete pstmt;
	return attr;
}

std::vector<ModelListAttr> ModelList::getResult(sql::ResultSet* res)
{
	std::vector<ModelListAttr> ret;
	if (res == nullptr) {
		std::cout << "ModelList::getResult: ResultSet is null!" << std::endl;
		return ret;
	}
	while (res->next()) {
		ModelListAttr attr;
		attr.id = res->getString("id");
		attr.modelname = res->getString("modelname");
		attr.input = res->getString("input");
		attr.output = res->getString("output");
		attr.output_size = res->getString("output_size");
		attr.task = res->getString("task");
		ret.emplace_back(attr);
	}
	return ret;
}