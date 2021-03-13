#pragma once

#ifndef _OPCDR_INTERFACE_H_
#define _OPCDR_INTERFACE_H_

#ifdef _OPCDR_EXPORT_
#define OPCDR_API extern "C" __declspec(dllexport)
#else 
#define OPCDR_API extern "C" __declspec(dllimport)
#endif // !TenCExport
#include "anno.h"
typedef void* JavaHandle;

struct MyPoint {
	int x;
	int y;
};

OPCDR_API JavaHandle initialize_handle(const char* ini_path);
OPCDR_API void run(JavaHandle handle, Anno* annos, MyPoint *topLeft, const char* ini_path, const char* slide, int recom_num, const char* savePath);
//OPCDR_API void run2(Anno* annos, const char* ini_path, const char* slide, int recom_num, const char* savePath);
OPCDR_API void freeModelMem(JavaHandle handle);

OPCDR_API void setCudaVisibleDevices(const char* num);
OPCDR_API void setAdditionalPath(const char* path);

#endif