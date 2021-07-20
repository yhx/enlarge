
#ifndef CATCH_H
#define CATCH_H

#include "catch.hpp"

#define TEST(a, b) TEST_CASE(#b, #a)

#define EQUALS(val) Catch::Equals(vector<int>((val)))
#define EQUALT(val) Catch::Equals(vector<Type>((val)))
#define APPROX(val) Catch::Approx(vector<real>((val)))

#define ASSERT_TRUE(a) REQUIRE((a))
#define ASSERT_EQ(a, b) REQUIRE((a) == (b))
#define ASSERT_LT(a, b) REQUIRE((a) < (b))

#define ASSERT_THAT REQUIRE_THAT

#define CATCH_MAIN int ret = Catch::Session().run(argc, argv)
#endif // CATCH_H
