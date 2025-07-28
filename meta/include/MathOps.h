//
// Created by Gary27 on 2025/3/20.
//

#ifndef HIP_RUNTIME_MATHOPS_H
#define HIP_RUNTIME_MATHOPS_H

#include "TypeTraits.h"
#include "StaticCast.h"

using detail::TypeTraits;

// Metavariable to check if two types are compound and have the same number of components.
template<class T, class U, class = Require<HasTypeTraits<T, U>>>
constexpr bool IsSameCompound = IsCompound<T> && TypeTraits<T>::components == TypeTraits<U>::components;

// Metavariable to check that at least one type is of compound type out of two types.
// If both are compound type, then it is checked that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompound =
        (TypeTraits<T>::components == 0 && TypeTraits<U>::components >= 1) ||
        (TypeTraits<T>::components >= 1 && TypeTraits<U>::components == 0) ||
        IsSameCompound<T, U>;

// Metavariable to check if a type is of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegral = std::is_integral_v<typename TypeTraits<T>::base_type>;

// Metavariable to require that at least one type is of compound type out of two integral types.
// If both are compound type, then it is required that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompoundAndBothAreIntegral = OneIsCompound<T, U> && IsIntegral<T> && IsIntegral<U>;

// Metavariable to require that a type is a CUDA compound of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegralCompound = IsIntegral<T> && IsCompound<T>;



// NVCV_CUDA_UNARY_OPERATOR

//template<typename T, class = Require<IsCompound<T>>>
//inline __host__ __device__ auto operator -(T a){
//
//}


// NVCV_CUDA_BINARY_OPERATOR

//template<typename T, typename U, class = Require<OneIsCompound<T, U>>>
//inline __host__ __device__ auto operator/(T a, U b){
//    using RT = MakeType<
//            decltype(std::declval<BaseType<T>>() / std::declval<BaseType<U>>()),  // cuda中找出基础类型互操作的结果类型
//            NumComponents<T> == 0 ? NumComponents<U> : NumComponents<T>
//            >;
//    if constexpr (NumComponents<T> == 0){
//        // a: float / b: float1
//        if constexpr (NumElements<RT> == 1) return RT{a / b.x};
//        else if constexpr (NumElements<RT> == 2) return RT{a / b.x, a / b.y};
//        else if constexpr (NumComponents<RT> == 3) return RT{a / b.x, a / b.y, a / b.z};
//        else if constexpr (NumElements<RT> == 4) return RT{a / b.x, a / b.y, a / b.z, a / b.w};
//    } else if constexpr (NumComponents<U> == 0){
//        if constexpr (NumElements<RT> == 1) return RT{a.x / b};
//        else if constexpr (NumElements<RT> == 2) return RT{a.x / b, a.y / b};
//        else if constexpr (NumComponents<RT> == 3) return RT{a.x / b, a.y / b, a.z / b};
//        else if constexpr (NumElements<RT> == 4) return RT{a.x / b, a.y / b, a.z / b, a.w / b};
//    } else {
//        if constexpr (NumElements<RT> == 1) return RT{ a.x / b.x};
//        else if constexpr (NumElements<RT> == 2) return RT{ a.x / b.x, a.y / b.y};
//        else if constexpr (NumElements<RT> == 3) return RT{ a.x / b.x, a.y / b.y, a.z / b.z};
//        else if constexpr (NumElements<RT> == 24) return RT{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
//    }
//}


//template<typename T, typename U, unsigned int tn, unsigned int un, class = Require<IsCompound<T> && IsCompound<U>>>
//__HOST_DEVICE__ inline constexpr auto operator/(const HIP_vector_type<T, tn>& a, const HIP_vector_type<U, un>& b){
//
//}


//template<typename T, typename U, unsigned int n>
//__HOST_DEVICE__ inline constexpr auto operator/(const HIP_vector_type<T, n>& a, const HIP_vector_type<U, n>& b){
//    using RT = HIP_vector_type<decltype(std::declval<T>() / std::declval<U>()), n>;
//    if constexpr (n == 1) return RT{a.x / b.x};
//    else if constexpr (n == 2) return RT{ a.x / b.x, a.y / b.y};
//    else if constexpr (n == 3) return RT{ a.x / b.x, a.y / b.y, a.z / b.z};
//    else if constexpr (n == 4) return RT{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w};
//}



#define NVCV_CUDA_BINARY_OPERATOR(OPERATOR, REQUIREMENT)                                                        \
    template<typename T, typename U, class = Require<REQUIREMENT<T, U>>>                            \
    inline __host__ __device__ auto operator OPERATOR(T a, U b)                                                 \
    {                                                                                                           \
        using RT = MakeType<                                                                        \
            decltype(std::declval<BaseType<T>>() OPERATOR std::declval<BaseType<U>>()), \
            NumComponents<T> == 0 ? NumComponents<U> : NumComponents<T>>;   \
        if constexpr (NumComponents<T> == 0)                                                        \
        {                                                                                                       \
            if constexpr (NumElements<RT> == 1)                                                     \
                return RT{a OPERATOR b.x};                                                                      \
            else if constexpr (NumElements<RT> == 2)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y};                                                      \
            else if constexpr (NumElements<RT> == 3)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z};                                      \
            else if constexpr (NumElements<RT> == 4)                                                \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z, a OPERATOR b.w};                      \
        }                                                                                                       \
        else if constexpr (NumComponents<U> == 0)                                                   \
        {                                                                                                       \
            if constexpr (NumElements<RT> == 1)                                                     \
                return RT{a.x OPERATOR b};                                                                      \
            else if constexpr (NumElements<RT> == 2)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b};                                                      \
            else if constexpr (NumElements<RT> == 3)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b};                                      \
            else if constexpr (NumElements<RT> == 4)                                                \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b, a.w OPERATOR b};                      \
        }                                                                                                       \
        else                                                                                                    \
        {                                                                                                       \
            if constexpr (NumElements<RT> == 1)                                                     \
                return RT{a.x OPERATOR b.x};                                                                    \
            else if constexpr (NumElements<RT> == 2)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y};                                                  \
            else if constexpr (NumElements<RT> == 3)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                                \
            else if constexpr (NumElements<RT> == 4)                                                \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z, a.w OPERATOR b.w};              \
        }                                                                                                       \
    }                                                                                                           \
    template<typename T, typename U, class = Require<IsCompound<T>>>                    \
    inline __host__ __device__ T &operator OPERATOR##=(T &a, U b)                                               \
    {                                                                                                           \
        return a = StaticCast<BaseType<T>>(a OPERATOR b);                               \
    }

NVCV_CUDA_BINARY_OPERATOR(-, OneIsCompound);
NVCV_CUDA_BINARY_OPERATOR(+, OneIsCompound);
NVCV_CUDA_BINARY_OPERATOR(*, OneIsCompound);
NVCV_CUDA_BINARY_OPERATOR(/, OneIsCompound);
NVCV_CUDA_BINARY_OPERATOR(%, OneIsCompoundAndBothAreIntegral);
NVCV_CUDA_BINARY_OPERATOR(&, OneIsCompoundAndBothAreIntegral);
NVCV_CUDA_BINARY_OPERATOR(|, OneIsCompoundAndBothAreIntegral);
NVCV_CUDA_BINARY_OPERATOR(^, OneIsCompoundAndBothAreIntegral);
NVCV_CUDA_BINARY_OPERATOR(<<,OneIsCompoundAndBothAreIntegral);
NVCV_CUDA_BINARY_OPERATOR(>>,OneIsCompoundAndBothAreIntegral);

#undef NVCV_CUDA_BINARY_OPERATOR


// 这里边的declval不能用，得想办法写一个declval。。。用来生成







#endif //HIP_RUNTIME_MATHOPS_H
