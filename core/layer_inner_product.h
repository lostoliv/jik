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


#ifndef CORE_LAYER_INNER_PRODUCT_H_
#define CORE_LAYER_INNER_PRODUCT_H_


#include <core/layer.h>
#include <core/log.h>
#include <core/rand.h>
#include <memory>
#include <vector>
#include <cmath>


namespace jik {


/*!
 *  \class  LayerInnerProduct
 *  \brief  Inner Product (aka fully connected) layer
 */
template <typename Dtype>
class LayerInnerProduct: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  LayerInnerProduct(const char*                                     name,
                    const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
                    const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Parameters
    bool use_bias;
    param.Get("use_bias", true, &use_bias);
    uint32_t num_output;
    param.Get("num_output", &num_output);

    // Input size
    uint32_t num_input = Parent::in_[0]->size[0] * Parent::in_[0]->size[1] *
                         Parent::in_[0]->size[2];

    // Create 2 weights: kernel filter and bias
    // Initialize the filter matrix with some random values
    // (gaussian distribution)
    Parent::weight_.resize(use_bias ? 2 : 1);
    Parent::weight_[0] = Rand<Dtype>::GenMatGauss(
      num_input, num_output, 1, 1, Dtype(0), std::sqrt(Dtype(1) / num_input));

    // Create the bias and initialize it to 0
    if (use_bias) {
      Parent::weight_[1] = std::make_shared<Mat<Dtype>>(1, 1, num_output);
    }

    // Create 1 output
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(
      1, 1, num_output, Parent::in_[0]->size[3]);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerInnerProduct() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data    = Parent::out_[0]->Data();
    const Dtype* in_data     = Parent::in_[0]->Data();
    const Dtype* filter_data = Parent::weight_[0]->Data();
    const Dtype* bias_data   = (Parent::weight_.size() > 1) ?
                               Parent::weight_[1]->Data() : nullptr;

    uint32_t num_out   = Parent::out_[0]->size[2];
    uint32_t num_in    = Parent::weight_[0]->size[0];
    uint32_t num_batch = Parent::in_[0]->size[3];

    // out = filter * in + bias
    for (uint32_t batch = 0; batch < num_batch; ++batch) {
      uint32_t in_offset  = num_in * batch;
      uint32_t out_offset = num_out * batch;
      for (uint32_t i = 0; i < num_out; ++i) {
        Dtype ip = Dtype(0);
        for (uint32_t j = 0; j < num_in; ++j) {
          uint32_t filter_index = num_in * i + j;
          ip += in_data[in_offset + j] * filter_data[filter_index];
        }
        out_data[out_offset + i] = ip;
        if (bias_data) {
          out_data[out_offset + i] += bias_data[i];
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
    const Dtype* out_grad_data    = Parent::out_[0]->Grad();
    const Dtype* in_data          = Parent::in_[0]->Data();
    const Dtype* filter_data      = Parent::weight_[0]->Data();
    Dtype*       in_grad_data     = Parent::in_[0]->Grad();
    Dtype*       filter_grad_data = Parent::weight_[0]->Grad();
    Dtype*       bias_grad_data   = (Parent::weight_.size() > 1) ?
                                    Parent::weight_[1]->Grad() : nullptr;

    uint32_t num_out   = Parent::out_[0]->size[2];
    uint32_t num_in    = Parent::weight_[0]->size[0];
    uint32_t num_batch = Parent::in_[0]->size[3];

    // in_grad     = filter * out_grad
    // filter_grad = in * out_grad
    // bias_grad   = out_grad
    for (uint32_t batch = 0; batch < num_batch; ++batch) {
      uint32_t in_offset  = num_in  * batch;
      uint32_t out_offset = num_out * batch;
      for (uint32_t i = 0; i < num_out; ++i) {
        Dtype dv = out_grad_data[out_offset + i];
        for (uint32_t j = 0; j < num_in; ++j) {
          uint32_t filter_index           = num_in * i + j;
          in_grad_data[in_offset + j]    += dv * filter_data[filter_index];
          filter_grad_data[filter_index] += dv * in_data[in_offset + j];
        }
        if (bias_grad_data) {
          bias_grad_data[i] += dv;
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_INNER_PRODUCT_H_
