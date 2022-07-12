
#ifndef CATCH_H
#define CATCH_H

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#define TEST(a, b) TEST_CASE(#b, #a)

#define EQUALS(...) Catch::Equals(vector<int>(__VA_ARGS__))
#define EQUALT(...) Catch::Equals(vector<Type>(__VA_ARGS__))
#define APPROX(...) Catch::Approx(vector<real>(__VA_ARGS__))

#define ASSERT_TRUE(a) REQUIRE((a))
#define ASSERT_EQ(a, b) REQUIRE((a) == (b))
#define ASSERT_FLOAT_EQ(a, b) REQUIRE((a) == Catch::Detail::Approx( b ))
#define ASSERT_LT(a, b) REQUIRE((a) < (b))

#define ASSERT_THAT REQUIRE_THAT

#define CATCH_RET() Catch::Session().run(argc, argv)

#define CATCH_MAIN int main(int argc, char **argc) \
{ \
	return CATCH_RET(); \
}

#endif // CATCH_H
