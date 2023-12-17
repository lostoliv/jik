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


#ifndef CORE_MAT_H_
#define CORE_MAT_H_


#include <vector>
#include <memory>
#include <cstring>


namespace jik {


/*!
 *  \class  Mat
 *  \brief  Matrix class (aka Tensor)
 *
 * This class is used to save the data or model weights.
 * It also stores (only if needed) the partial gradients (Jacobian).
 *
 * We do not support matrices/tensors of arbitrary shapes in order to not make
 * the code too complex. As-is, this is mostly designed for images-based tasks.
 *
 * The matrix has 4 dimensions n*m*d*b:
 *  + n can be interpreted as the number of columns (or width)
 *  + m can be interpreted as the number of rows (or height)
 *  + d can be interpreted as the depth
 *  + b can be interpreted as the batch size
 *
 * For example, a matrix representing 10 640x480 RGB images
 * will have its dimensions set to 640*480*3*10.
 *
 * This type of data structure is also called tensor or multidimensional array
 * in some framework. The fact that we are setting the dimensions to 4 is to
 * allow implicit reshaping when going from fully-connected layers to
 * convolution layers (or vice versa).
 */
template <typename Dtype>
class Mat {
  // Public types
 public:
  typedef Dtype Type;


  // Public attributes
 public:
  uint32_t                    size[4];  // Matrix size
  std::vector<Dtype>          data;     // Matrix data
  std::shared_ptr<Mat<Dtype>> grad;     // Gradients matrix (Jacobian)


  // Public methods
 public:
  /*!
   * Default constructor.
   */
  Mat() {
    size[0] = size[1] = size[2] = size[3] = 0;
  }

  /*!
   * Constructor.
   *
   *  \param[in]  n        : matrix size (number of columns)
   *  \param[in]  m        : matrix size (number of rows)
   *  \param[in]  d        : matrix size (depth)
   *  \param[in]  b        : matrix size (batch size)
   *  \param[in]  init_grad: initialize the gradients matrix?
   */
  Mat(uint32_t n, uint32_t m, uint32_t d = 1, uint32_t b = 1,
      bool init_grad = true) {
    size[0] = n;
    size[1] = m;
    size[2] = d;
    size[3] = b;
    data.resize(size[0] * size[1] * size[2] * size[3], Dtype(0));
    if (init_grad) {
      grad = std::make_shared<Mat<Dtype>>(size, false);
    }
  }

  /*!
   * Constructor.
   *
   *  \param[in]  sz       : matrix size
   *  \param[in]  init_grad: initialize the gradients matrix?
   */
  explicit Mat(const uint32_t* sz, bool init_grad = true):
    Mat(sz[0], sz[1], sz[2], sz[3], init_grad) {}

  /*!
   * Destructor.
   */
  ~Mat() {}

  /*!
   * Get the data (const).
   *
   *  \return Data
   */
  const Dtype* Data() const {
    if (data.empty()) {
      return nullptr;
    }
    return &data[0];
  }

  /*!
   * Get the data.
   *
   *  \return Data
   */
  Dtype* Data() {
    if (data.empty()) {
      return nullptr;
    }
    return &data[0];
  }

  /*!
   * Get the gradients (const).
   *
   *  \return Gradients
   */
  const Dtype* Grad() const {
    if (!grad || grad->data.empty()) {
      return nullptr;
    }
    return &grad->data[0];
  }

  /*!
   * Get the gradients.
   *
   *  \return Gradients
   */
  Dtype* Grad() {
    if (!grad || grad->data.empty()) {
      return nullptr;
    }
    return &grad->data[0];
  }

  /*!
   * Get the matrix size (1D).
   *
   *  \return Matrix size (1D)
   */
  uint32_t Size() const {
    return uint32_t(data.size());
  }

  /*!
   * Set the matrix to a special value.
   *
   *  \param  val: value
   */
  void Set(Dtype val) {
    std::fill(data.begin(), data.end(), val);
  }

  /*!
   * Zero out the matrix.
   */
  void Zero() {
    std::memset(&data[0], 0, data.size() * sizeof(Dtype));
  }

  /*!
   * Zero out the gradients matrix.
   */
  void ZeroGrad() {
    if (grad) {
      std::memset(&grad->data[0], 0, grad->data.size() * sizeof(Dtype));
    }
  }
};


}  // namespace jik


#endif  // CORE_MAT_H_
