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


#ifndef CORE_LAYER_MULT_H_
#define CORE_LAYER_MULT_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  LayerMult
 *  \brief  Matrices multiplication
 */
template <typename Dtype>
class LayerMult: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name: layer name
   *  \param[in]  in  : input activations
   */
  LayerMult(const char*                                     name,
            const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {
    // Make sure we have 2 inputs and they have compatible sizes
    Check(Parent::in_.size() == 2, "Layer '%s' must have 2 inputs",
          Parent::Name());
    Check(Parent::in_[0]->size[1] == Parent::in_[1]->size[0] &&
          Parent::in_[0]->size[2] == Parent::in_[1]->size[2] &&
          Parent::in_[0]->size[3] == Parent::in_[1]->size[3],
          "Layer '%s' inputs must have compatible sizes", Parent::Name());

    // Create 1 output
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(
      Parent::in_[0]->size[0], Parent::in_[1]->size[1],
      Parent::in_[0]->size[2], Parent::in_[0]->size[3]);
  }

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data = Parent::out_[0]->Data();
    const Dtype* in1_data = Parent::in_[0]->Data();
    const Dtype* in2_data = Parent::in_[1]->Data();

    uint32_t m = Parent::in_[0]->size[0];
    uint32_t n = Parent::in_[1]->size[1];
    uint32_t k = Parent::in_[1]->size[0];

    uint32_t in1_size = Parent::in_[0]->size[0]  * Parent::in_[0]->size[1];
    uint32_t in2_size = Parent::in_[1]->size[0]  * Parent::in_[1]->size[1];
    uint32_t out_size = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];

    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];

    // out = in1 * in2
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        uint32_t offset     = channel + num_channel * batch;
        uint32_t in1_offset = offset * in1_size;
        uint32_t in2_offset = offset * in2_size;
        uint32_t out_offset = offset * out_size;
        for (uint32_t i = 0, ia = 0, ic = 0; i < m; ++i, ia += k, ic += n) {
          for (uint32_t j = 0, jb = 0, jc = 0; j < n; ++j, ++jb, ++jc) {
            uint32_t index  = out_offset + jc + ic;
            out_data[index] = Dtype(0);
            for (uint32_t l = 0, la = 0, lb = 0; l < k; ++l, ++la, lb += n) {
              out_data[index] += in1_data[in1_offset + la + ia] *
                                 in2_data[in2_offset + lb + jb];
            }
          }
        }
      }
    }
  }

  /*!
   * Backward pass.
   * The backward pass calculates the inputs activations and weights
   * gradients in regard to the outputs activations gradients.
   *
   *  \param[in]  state: state
   */
  virtual void Backward(const State& state) {
    const Dtype* out_grad_data = Parent::out_[0]->Grad();
    const Dtype* in1_data      = Parent::in_[0]->Data();
    const Dtype* in2_data      = Parent::in_[1]->Data();
    Dtype*       in1_grad_data = Parent::in_[0]->Grad();
    Dtype*       in2_grad_data = Parent::in_[1]->Grad();

    uint32_t m = Parent::in_[0]->size[0];
    uint32_t n = Parent::in_[1]->size[1];
    uint32_t k = Parent::in_[0]->size[1];

    uint32_t in1_size = Parent::in_[0]->size[0]  * Parent::in_[0]->size[1];
    uint32_t in2_size = Parent::in_[1]->size[0]  * Parent::in_[1]->size[1];
    uint32_t out_size = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];

    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];

    // in1_grad = in2 * out_grad
    // in2_grad = in1 * out_grad
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        uint32_t offset     = channel + num_channel * batch;
        uint32_t in1_offset = offset * in1_size;
        uint32_t in2_offset = offset * in2_size;
        uint32_t out_offset = offset * out_size;
        for (uint32_t i = 0; i < m; ++i) {
          for (uint32_t j = 0; j < n; ++j) {
            Dtype dv = out_grad_data[out_offset + n * i + j];
            for (uint32_t l = 0; l < k; ++l) {
              uint32_t i1 = in1_offset + k * i + l;
              uint32_t i2 = in2_offset + n * l + j;
              in1_grad_data[i1] += in2_data[i2] * dv;
              in2_grad_data[i2] += in1_data[i1] * dv;
            }
          }
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_MULT_H_
