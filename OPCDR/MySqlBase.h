#pragma once
#ifndef _OPCDR_MYSQLBASE_H_
#define _OPCDR_MYSQLBASE_H_

/* Standard C++ includes */
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>

/*
  Include directly the different
  headers from cppconn/ and mysql_driver.h + mysql_util.h
  (and mysql_connection.h). This will reduce your build time!
*/
#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>

template <typename TableAttr>
class MySqlBase
{
public:
    sql::Driver* driver;
    sql::Connection* con;

public:
    std::string schema = "das";

public:
    MySqlBase() = delete;
    MySqlBase(std::string hostName, std::string userName, std::string password);
    virtual ~MySqlBase();

    virtual std::vector<TableAttr> getResult(sql::ResultSet* res) = 0;
    std::vector<TableAttr> selectRows(std::string sql);

    //下下策，只能将ResultSet*交给下面的子类处理了，太模板的函数现在还想不出来~（而且返回指针实在是不好的做法）
    sql::ResultSet* select(std::string sql);
    sql::ResultSet* select(sql::PreparedStatement* pstmt);
    bool update(std::string sql);
    bool update(sql::PreparedStatement* pstmt);
    void exception_handle(sql::SQLException& e);
};

template <typename TableAttr>
MySqlBase<TableAttr>::MySqlBase(std::string hostName, std::string userName, std::string password)
{
    driver = get_driver_instance();
    con = driver->connect(hostName, userName, password);
    con->setSchema(schema);
}

template <typename TableAttr>
MySqlBase<TableAttr>::~MySqlBase()
{
    delete con;
}

template <typename TableAttr>
sql::ResultSet* MySqlBase<TableAttr>::select(sql::PreparedStatement* pstmt)
{
    sql::ResultSet* res = nullptr;
    try {
        res = pstmt->executeQuery();
    }
    catch (sql::SQLException & e) {
        exception_handle(e);
    }
    return res;
}

template <typename TableAttr>
sql::ResultSet* MySqlBase<TableAttr>::select(std::string sql)
{
    using namespace std;
    sql::ResultSet* res = nullptr;
    try {
        sql::Statement* stmt;
        stmt = con->createStatement();
        res = stmt->executeQuery(sql);
        delete stmt;
    }
    catch (sql::SQLException & e) {
        exception_handle(e);
    }
    return res;
}

template <typename TableAttr>
std::vector<TableAttr> MySqlBase<TableAttr>::selectRows(std::string sql)
{
    using namespace std;
    std::vector<TableAttr> ret;
    try {
        sql::ResultSet* res;
        sql::Statement* stmt;
        stmt = con->createStatement();
        res = stmt->executeQuery(sql);
        ret = getResult(res);

        delete res;
        delete stmt;
        return ret;
    }
    catch (sql::SQLException & e) {
        exception_handle(e);
    }
    return ret;
}

template <typename TableAttr>
bool MySqlBase<TableAttr>::update(std::string sql)
{
    using namespace std;
    try {
        sql::Statement* stmt;
        stmt = con->createStatement();
        stmt->executeUpdate(sql);
        delete stmt;
        return true;
    }
    catch (sql::SQLException & e) {
        exception_handle(e);
        return false;
    }
}

template <typename TableAttr>
bool MySqlBase<TableAttr>::update(sql::PreparedStatement* pstmt)
{
    using namespace std;
    try {
        pstmt->executeUpdate();
        return true;
    }
    catch (sql::SQLException & e) {
        exception_handle(e);
        return false;
    }
}

template <typename TableAttr>
void MySqlBase<TableAttr>::exception_handle(sql::SQLException& e)
{
    using namespace std;
    cout << "# ERR: SQLException in " << __FILE__;
    cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
    cout << "# ERR: " << e.what();
    cout << " (MySQL error code: " << e.getErrorCode();
    cout << ", SQLState: " << e.getSQLState() << " )" << endl;
}

#endif