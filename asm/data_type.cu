//
// Created by Gary27 on 2025/3/18.
//

#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>      // for FLT_MIN, etc.
#include <climits>     // for CHAR_MIN, etc.
#include <type_traits> // for std::remove_const, etc.

// Metatype to add information to regular C types and CUDA compound types.
template<class T>
struct TypeTraits;

#define NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE, BASE_TYPE, COMPONENTS, ELEMENTS, MIN_VAL, MAX_VAL) \
    template<>                                                                                  \
    struct TypeTraits<COMPOUND_TYPE>                                                            \
    {                                                                                           \
        using base_type                       = BASE_TYPE;                                      \
        static constexpr int       components = COMPONENTS;                                     \
        static constexpr int       elements   = ELEMENTS;                                       \
        static constexpr char      name[]     = #COMPOUND_TYPE;                                 \
        static constexpr base_type min        = MIN_VAL;                                        \
        static constexpr base_type max        = MAX_VAL;                                        \
    }

NVCV_CUDA_TYPE_TRAITS(dim3, unsigned int, 3, 3, 0, UINT_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned char, unsigned char, 0, 1, 0, UCHAR_MAX);

NVCV_CUDA_TYPE_TRAITS(signed char, signed char, 0, 1, SCHAR_MIN, SCHAR_MAX);
#if CHAR_MIN == 0
NVCV_CUDA_TYPE_TRAITS(char, unsigned char, 0, 1, 0, UCHAR_MAX);
#else
NVCV_CUDA_TYPE_TRAITS(char, signed char, 0, 1, SCHAR_MIN, SCHAR_MAX);
#endif
NVCV_CUDA_TYPE_TRAITS(short, short, 0, 1, SHRT_MIN, SHRT_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned short, unsigned short, 0, 1, 0, USHRT_MAX);

NVCV_CUDA_TYPE_TRAITS(int, int, 0, 1, INT_MIN, INT_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned int, unsigned int, 0, 1, 0, UINT_MAX);

NVCV_CUDA_TYPE_TRAITS(long, long, 0, 1, LONG_MIN, LONG_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned long, unsigned long, 0, 1, 0, ULONG_MAX);

NVCV_CUDA_TYPE_TRAITS(long long, long long, 0, 1, LLONG_MIN, LLONG_MAX);

NVCV_CUDA_TYPE_TRAITS(unsigned long long, unsigned long long, 0, 1, 0, ULLONG_MAX);

NVCV_CUDA_TYPE_TRAITS(float, float, 0, 1, FLT_MIN, FLT_MAX);

NVCV_CUDA_TYPE_TRAITS(double, double, 0, 1, DBL_MIN, DBL_MAX);

#define NVCV_CUDA_TYPE_TRAITS_1_TO_4(COMPOUND_TYPE, BASE_TYPE, MIN_VAL, MAX_VAL) \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##1, BASE_TYPE, 1, 1, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##2, BASE_TYPE, 2, 2, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##3, BASE_TYPE, 3, 3, MIN_VAL, MAX_VAL);  \
    NVCV_CUDA_TYPE_TRAITS(COMPOUND_TYPE##4, BASE_TYPE, 4, 4, MIN_VAL, MAX_VAL)

NVCV_CUDA_TYPE_TRAITS_1_TO_4(char, signed char, SCHAR_MIN, SCHAR_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(uchar, unsigned char, 0, UCHAR_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(short, short, SHRT_MIN, SHRT_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(ushort, unsigned short, 0, USHRT_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(int, int, INT_MIN, INT_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(uint, unsigned int, 0, UINT_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(long, long, LONG_MIN, LONG_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(ulong, unsigned long, 0, ULONG_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(longlong, long long, LLONG_MIN, LLONG_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(ulonglong, unsigned long long, 0, ULLONG_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(float, float, FLT_MIN, FLT_MAX);

NVCV_CUDA_TYPE_TRAITS_1_TO_4(double, double, DBL_MIN, DBL_MAX);

#undef NVCV_CUDA_TYPE_TRAITS_1_TO_4
#undef NVCV_CUDA_TYPE_TRAITS

template<typename T, typename = void>
struct HasTypeTraits_t : std::false_type {
};

template<typename T>
struct HasTypeTraits_t<T, std::void_t<typename TypeTraits<T>::base_type> > : std::true_type {
};

// Metavariable to check if one or more types have type traits.
template<typename... Ts>
constexpr bool HasTypeTraits = (HasTypeTraits_t<Ts>::value && ...);


// template<typename T1, typename T2>
// __device__ void func() {
//     using IntermediateT = decltype(std::declval<T1>() * std::declval<T2>());
// }
//
//
// __global__ void func1() {
//     func<float, uchar1>();
// }

template <typename T, typename U>
struct MultiplyResult {
    // 使用 decltype 和 std::declval 来推导 T * U 的返回类型
    using IntermediateT = decltype(std::declval<T>() * std::declval<U>());
    // using IntermediateT = decltype(T{} * U{});

    static_assert(std::is_same_v<IntermediateT, uchar1>);
};

template <typename T, typename U>
using MultiplyResultT = typename MultiplyResult<T, U>::IntermediateT;


template<typename T1, typename... T2>
void type_check() {
    // 使用折叠表达式展开 T2 参数包
    static_assert((std::is_same_v<T1, T2> || ...), "TYPE must be one of the specified types");
}


int main() {

    float2 i2 {2, 2};

    float2 f2 {4., 4.};


    auto res1 = i2 / f2;
    std::cout << "res1.x: " << res1.x << std::endl;

    long double a;

    type_check<float, float, double, int>();  // 通过

    // // 测试 float 和 int 相乘的结果类型
    // using ResultType = MultiplyResultT<float, uchar1>;
    //
    // // 打印结果类型
    // std::cout << "Result type of float * char1: "
    //           << typeid(ResultType).name() << std::endl;

    // func1<<<1,1>>>();
    // cudaDeviceSynchronize();

    std::cout << "Type Size -----: " << std::endl;
    std::cout << "sizeof(char): " << sizeof(char) << std::endl;
    std::cout << "sizeof(unsigned char): " << sizeof(unsigned char) << std::endl;
    std::cout << "sizeof(short): " << sizeof(short) << std::endl;
    std::cout << "sizeof(unsigned short): " << sizeof(unsigned short) << std::endl;
    std::cout << "sizeof(int): " << sizeof(int) << std::endl;
    std::cout << "sizeof(unsigned int): " << sizeof(unsigned int) << std::endl;
    std::cout << "sizeof(long): " << sizeof(long) << std::endl;
    std::cout << "sizeof(unsigned long): " << sizeof(unsigned long) << std::endl;
    std::cout << "sizeof(long long): " << sizeof(long long) << std::endl;
    std::cout << "sizeof(unsigned long long): " << sizeof(unsigned long long) << std::endl;
    std::cout << "sizeof(float): " << sizeof(float) << std::endl;
    std::cout << "sizeof(double): " << sizeof(double) << std::endl;
    std::cout << "sizeof(long double): " << sizeof(long double) << std::endl;



    int3 i3;

    char1 c1;
    short1 s1;


    if (std::numeric_limits<char>::is_signed) {
        std::cout << "char is signed on this platform." << std::endl;
    } else {
        std::cout << "char is unsigned on this platform." << std::endl;
    }

    if constexpr (HasTypeTraits<char1>) {
        std::cout << "char1 has type traits" << '\n';
    } else {
        std::cout << "No type traits" << '\n';
    }

    std::cout << sizeof(char1) << std::endl;

    // static_assert(std::is_same_v<char1, char>);

    // char1 --> signed char

    char1 c{1};
    static_assert(std::is_same_v<decltype(c.x), signed char>, "qual");
    std::cout << sizeof(char1) << std::endl;
    std::cout << "char1: " << c.x << std::endl;

    return 0;
}
