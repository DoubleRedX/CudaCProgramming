//
// Created by Gary27 on 2025/3/20.
//

#ifndef HIP_RUNTIME_TYPETRAITS_H
#define HIP_RUNTIME_TYPETRAITS_H

#include "MetaProgramming.h"

// Metatype to serve as a requirement for a template object to meet the given boolean expression.
template<bool B>
using Require = std::enable_if_t<B>;

// Metavariable to check if one or more types have type traits.
template<typename... Ts>
constexpr bool HasTypeTraits = (detail::HasTypeTraits_t<Ts>::value && ...);

// Metavariable to check if a type is a CUDA compound type.
template<class T, class = Require<HasTypeTraits<T>>>
constexpr bool IsCompound = detail::TypeTraits<T>::components >= 1;

// Metavariable to check if a CUDA compound type T has N or more components.
template<typename T, int N, class = Require<HasTypeTraits<T>>>
constexpr bool HasEnoughComponents = N <= detail::TypeTraits<T>::components;

/**
 * Metatype to get the base type of a CUDA compound types.
 *
 * @code
 * using DataType = ...;
 * using ChannelType = nvcv::cuda::BaseType<DataType>;
 * @endcode
 *
 * @note This is identity for regular C types.
 *
 * @tparam T Type to get the base type from.
 */
template<class T, class = Require<HasTypeTraits<T>>>
using BaseType = typename detail::TypeTraits<T>::base_type;

/**
 * Metavariable to get the number of components of a type.
 *
 * @code
 * using DataType = ...;
 * int nc = nvcv::cuda::NumComponents<DataType>;
 * @endcode
 *
 * @note This is zero for regular C types.
 *
 * @tparam T Type to get the number of components from.
 */
template<class T, class = Require<HasTypeTraits<T>>>
constexpr int NumComponents = detail::TypeTraits<T>::components;

/**
 * Metavariable to get the number of elements of a type.
 *
 * @code
 * using DataType = ...;
 * for (int e = 0; e < nvcv::cuda::NumElements<DataType>; ++e)
 *     // ...
 * @endcode
 *
 * @note This is one for regular C types and one to four for CUDA compound types.
 *
 * @tparam T Type to get the number of elements from.
 */
template<class T, class = Require<HasTypeTraits<T>>>
constexpr int NumElements = detail::TypeTraits<T>::elements;

/**
 * Metatype to make a type from a base type and number of components.
 *
 * When number of components is zero, it yields the identity (regular C) type, and when it is between 1
 * and 4 it yields the CUDA compound type.
 *
 * @code
 * using RGB8Type = MakeType<unsigned char, 3>; // yields uchar3
 * @endcode
 *
 * @note Note that T=char might yield uchar1..4 types when char is equal unsigned char, i.e. CHAR_MIN == 0.
 *
 * @tparam T Base type to make the type from.
 * @tparam C Number of components to make the type.
 */
template<class T, int C, class = Require<HasTypeTraits<T>>>
using MakeType = detail::MakeType_t<T, C>;


/**
 * Metatype to convert the base type of a type.
 *
 * The base type of target type \p T is replaced to be \p BT.
 *
 * @code
 * using DataType = ...;
 * using FloatDataType = ConvertBaseTypeTo<float, DataType>; // yields float1..4
 * @endcode
 *
 * @tparam BT Base type to use in the conversion.
 * @tparam T Target type to convert its base type.
 */
template<class BT, class T, class = Require<HasTypeTraits<BT, T>>>
using ConvertBaseTypeTo = detail::ConvertBaseTypeTo_t<BT, T>;


/**
 * Metafunction to get an element by reference from a given value reference.
 *
 * The value may be of CUDA compound type with 1 to 4 elements, where the corresponding element index is 0
 * to 3, and the return is a reference to the element with the base type of the compound type, copying the
 * constness (that is the return reference is constant if the input value is constant).  The value may be a regular
 * C type, in which case the element index is ignored and the identity is returned.  It is a requirement of the
 * GetElement function that the type \p T has type traits.
 *
 * @code
 * using PixelRGB8Type = MakeType<unsigned char, 3>;
 * PixelRGB8Type pix = ...;
 * auto green = GetElement(pix, 1); // yields unsigned char
 * @endcode
 *
 * @tparam T Type of the value to get the element from.
 *
 * @param[in] v Value of type T to get an element from.
 * @param[in] eidx Element index in [0, 3] inside the compound value to get the reference from.
 *                 This element index is ignored in case the value is not of a CUDA compound type.
 *
 * @return The reference of the value's element.
 */
template<typename T,
        typename RT = detail::CopyConstness_t<T, std::conditional_t<IsCompound<T>, BaseType<T>, T>>,
class = Require<HasTypeTraits<T>>>
__host__ __device__ RT &GetElement(T &v, int eidx)
{
    if constexpr (IsCompound<T>)
    {
        assert(eidx < NumElements<T>);
        return reinterpret_cast<RT *>(&v)[eidx];
    }
    else
    {
        return v;
    }
}

template<int EIDX,
        typename T,
        typename RT = detail::CopyConstness_t<T, std::conditional_t<IsCompound<T>, BaseType<T>, T>>,
class       = Require<HasTypeTraits<T>>>
__host__ __device__ RT &GetElement(T &v)
{
    if constexpr (IsCompound<T>)
    {
        static_assert(EIDX < NumElements<T>);
        if constexpr (EIDX == 0)
        return v.x;
        else if constexpr (EIDX == 1)
        return v.y;
        else if constexpr (EIDX == 2)
        return v.z;
        else if constexpr (EIDX == 3)
        return v.w;
    }
    else
    {
        return v;
    }
}


#endif //HIP_RUNTIME_TYPETRAITS_H
