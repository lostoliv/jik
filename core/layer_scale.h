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


#ifndef CORE_LAYER_SCALE_H_
#define CORE_LAYER_SCALE_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  LayerScale
 *  \brief  Scale (and bias)
 */
template <typename Dtype>
class LayerScale: public Layer<Dtype> {
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
  LayerScale(const char*                                     name,
             const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
             const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    bool use_bias;
    param.Get("use_bias", true, &use_bias);

    // Create 2 weights: scale and bias
    // We learn 1 scale and bias per channel per image
    Parent::weight_.resize(use_bias ? 2 : 1);

    // Create the scale and initialize it to 1
    Parent::weight_[0] = std::make_shared<Mat<Dtype>>(
      1, 1, Parent::in_[0]->size[2]);
    Parent::weight_[0]->Set(Dtype(1));

    // Create the bias and initialize it to 0
    if (use_bias) {
      Parent::weight_[1] = std::make_shared<Mat<Dtype>>(
        1, 1, Parent::in_[0]->size[2]);
    }

    // Create 1 output, same size as the input
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerScale() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data   = Parent::out_[0]->Data();
    const Dtype* in_data    = Parent::in_[0]->Data();
    const Dtype* scale_data = Parent::weight_[0]->Data();
    const Dtype* bias_data  = (Parent::weight_.size() > 1) ?
                              Parent::weight_[1]->Data() : nullptr;

    uint32_t data_size   = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];
    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];

    // out = scale * in + bias
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        uint32_t offset = (batch * num_channel + channel) * data_size;
        for (uint32_t i = 0; i < data_size; ++i) {
          out_data[offset + i] = scale_data[channel] * in_data[offset + i];
          if (bias_data) {
            out_data[offset + i] += bias_data[channel];
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
    const Dtype* out_grad_data   = Parent::out_[0]->Grad();
    const Dtype* in_data         = Parent::in_[0]->Data();
    Dtype*       in_grad_data    = Parent::in_[0]->Grad();
    const Dtype* scale_data      = Parent::weight_[0]->Data();
    Dtype*       scale_grad_data = Parent::weight_[0]->Grad();
    Dtype*       bias_grad_data  = (Parent::weight_.size() > 1) ?
                                   Parent::weight_[1]->Grad() : nullptr;

    uint32_t data_size   = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];
    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];

    // in_grad    = scale * out_grad
    // scale_grad = in * out_grad
    // bias_grad  = out_grad
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        uint32_t offset = (batch * num_channel + channel) * data_size;
        for (uint32_t i = 0; i < data_size; ++i) {
          Dtype dv                  = out_grad_data[offset + i];
          in_grad_data[offset + i] += dv * scale_data[channel];
          scale_grad_data[channel] += dv * in_data[offset + i];
          if (bias_grad_data) {
            bias_grad_data[channel] += dv;
          }
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_SCALE_H_
