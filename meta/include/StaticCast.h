//
// Created by Gary27 on 2025/3/20.
//

#ifndef HIP_RUNTIME_STATICCAST_H
#define HIP_RUNTIME_STATICCAST_H

#include "TypeTraits.h"

/**
 * @defgroup NVCV_CPP_CUDATOOLS_STATICCAST Static Cast
 * @{
 */

/**
 * Metafunction to static cast all values of a compound to a target type.
 *
 * The template parameter \p T defines the base type (regular C type) to cast all components of the CUDA
 * compound type \p U passed as function argument \p u to the type \p T.  The static cast return type has the base
 * type \p T and the number of components as the compound type \p U.  For instance, an uint3 can be casted to int3
 * by passing it as function argument of StaticCast and the type int as template argument (see example below).  The
 * type \p U is not needed as it is inferred from the argument \u.  It is a requirement of the StaticCast function
 * that the type \p T is of regular C type and the type \p U is of CUDA compound type.
 *
 * @code
 * int3 idx = StaticCast<int>(blockIdx * blockDim + threadIdx);
 * @endcode
 *
 * @tparam T Type to do static cast on each component of \p u.
 *
 * @param[in] u Compound value to static cast each of its components to target type \p T.
 *
 * @return The compound value with all components static casted to type \p T.
 */
template<typename T, typename U, class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
__host__ __device__ auto StaticCast(U u)
{
    using RT = ConvertBaseTypeTo<T, U>;
    if constexpr (std::is_same_v<U, RT>)
    {
        return u;
    }
    else
    {
        RT out{};

        GetElement<0>(out) = static_cast<T>(GetElement<0>(u));
        if constexpr (NumElements<RT> >= 2)
        GetElement<1>(out) = static_cast<T>(GetElement<1>(u));
        if constexpr (NumElements<RT> >= 3)
        GetElement<2>(out) = static_cast<T>(GetElement<2>(u));
        if constexpr (NumElements<RT> == 4)
        GetElement<3>(out) = static_cast<T>(GetElement<3>(u));

        return out;
    }
}

#endif //HIP_RUNTIME_STATICCAST_H
