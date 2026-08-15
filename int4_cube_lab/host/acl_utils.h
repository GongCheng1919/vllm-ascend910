#pragma once
#include <iostream>
#include "acl/acl.h"

#define ACL_CHECK(expr) do { \
    aclError __e = (expr); \
    if (__e != ACL_SUCCESS) { \
        std::cerr << "ACL error " << __e \
                  << " at " << __FILE__ << ":" << __LINE__ \
                  << " in " << #expr << std::endl; \
        std::exit(1); \
    } \
} while (0)
