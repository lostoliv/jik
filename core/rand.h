/*!
  The MIT License (MIT)

  Copyright (c)2016 Olivier Soares

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
 */


#ifndef CORE_RAND_H_
#define CORE_RAND_H_


#include <core/mat.h>
#include <memory>
#include <random>


namespace jik {


/*!
 *  \struct Rand
 *  \brief  Random generator
 */
template <typename Dtype>
struct Rand {
  /*!
   * Generate a randomly distributed matrix.
   *
   *  \param[in]  n    : matrix size
   *  \param[in]  d    : matrix size
   *  \param[in]  m    : matrix size
   *  \param[in]  f    : matrix size
   *  \param[in]  low  : low boundary
   *  \param[in]  hight: high boundary
   *
   *  \return     Random matrix
   */
  static std::shared_ptr<Mat<Dtype>> GenMat(uint32_t n, uint32_t d,
                                            uint32_t m, uint32_t f,
                                            Dtype low, Dtype high) {
    std::shared_ptr<Mat<Dtype>> mat = std::make_shared<Mat<Dtype>>(n, d, m, f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Dtype> dist(low, high);

    Dtype* mat_data = mat->Data();
    for (uint32_t i = 0; i < mat->Size(); ++i) {
      mat_data[i] = dist(gen);
    }

    return mat;
  }


  /*!
   * Generate a randomly gaussian distributed matrix.
   *
   *  \param[in]  n      : matrix size
   *  \param[in]  d      : matrix size
   *  \param[in]  m      : matrix size
   *  \param[in]  f      : matrix size
   *  \param[in]  mean   : mean value
   *  \param[in]  std_dev: standard deviation
   *
   *  \return     Random matrix
   */
  static std::shared_ptr<Mat<Dtype>> GenMatGauss(uint32_t n, uint32_t d,
                                                 uint32_t m, uint32_t f,
                                                 Dtype mean, Dtype std_dev) {
    std::shared_ptr<Mat<Dtype>> mat = std::make_shared<Mat<Dtype>>(n, d, m, f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Dtype> dist(mean, std_dev);

    Dtype* mat_data = mat->Data();
    for (uint32_t i = 0; i < mat->Size(); ++i) {
      mat_data[i] = dist(gen);
    }

    return mat;
  }
};


}  // namespace jik


#endif  // CORE_RAND_H_
