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


#ifndef CORE_LAYER_CONV_H_
#define CORE_LAYER_CONV_H_


#include <core/layer.h>
#include <core/log.h>
#include <core/rand.h>
#include <memory>
#include <vector>
#include <cmath>


namespace jik {


/*!
 *  \class  LayerConv
 *  \brief  Matrix convolution
 */
template <typename Dtype>
class LayerConv: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Protected attributes
 protected:
  uint32_t num_output_;     // Number of outputs
  uint32_t filter_width_;   // Convolution kernel filter width
  uint32_t filter_height_;  // Convolution kernel filter height
  uint32_t padding_x_;      // Row padding
  uint32_t padding_y_;      // Column padding
  uint32_t stride_x_;       // Row stride
  uint32_t stride_y_;       // Column stride
  uint32_t out_width_;      // Output width
  uint32_t out_height_;     // Output height


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  LayerConv(const char*                                     name,
            const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
            const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Parameters
    bool use_bias;
    param.Get("use_bias"     , true, &use_bias);
    param.Get("num_output"   , &num_output_);
    param.Get("filter_width" , &filter_width_);
    param.Get("filter_height", &filter_height_);
    param.Get("padding_x"    , &padding_x_);
    param.Get("padding_y"    , &padding_y_);
    param.Get("stride_x"     , &stride_x_);
    param.Get("stride_y"     , &stride_y_);

    // Calculate the output width and height based on padding and stride
    out_width_ = (Parent::in_[0]->size[0] +
                 2 * padding_x_ - filter_width_)  / stride_x_ + 1;
    out_height_ = (Parent::in_[0]->size[1] +
                  2 * padding_y_ - filter_height_) / stride_y_ + 1;

    // Number of inputs
    uint32_t num_input = Parent::in_[0]->size[2];

    // Create 2 weights: kernel filter and bias
    Parent::weight_.resize(use_bias ? 2 : 1);

    // Initialize the filter matrix with some random values
    // (gaussian distribution)
    Parent::weight_[0] = Rand<Dtype>::GenMatGauss(
      filter_width_, filter_height_, num_input, num_output_, Dtype(0),
      std::sqrt(Dtype(1) / (filter_width_ * filter_height_ * num_input)));

    // Create the bias and initialize it to 0
    if (use_bias) {
      Parent::weight_[1] = std::make_shared<Mat<Dtype>>(1, 1, num_output_);
    }

    // Create 1 output, same size as the input
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(
      out_width_, out_height_, num_output_, Parent::in_[0]->size[3]);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerConv() {}

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

    uint32_t in_width   = Parent::in_[0]->size[0];
    uint32_t in_height  = Parent::in_[0]->size[1];
    uint32_t num_input  = Parent::in_[0]->size[2];
    uint32_t batch_size = Parent::in_[0]->size[3];

    // out = filter * in + bias
    // Notes: This is a fairly non-optimized way to perform convolution
    //        One trick is to linearize all the images of the mini-batch into a
    //        big 2D matrix (down from a 4D matrix), effectively converting one
    //        image into a 1D column, each row storing a different image from
    //        the mini-batch
    //        We also should be using a good linear algebra library (BLAS,
    //        Eigen, etc.) to perform the matrix multiplication: it will be
    //        much faster and padding/stride will be handled more efficiently
    //        All these optimizations are beyond the scope of this project but
    //        would be welcome additions
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t in_offset  = in_width   * in_height   * num_input   * batch;
      uint32_t out_offset = out_width_ * out_height_ * num_output_ * batch;
      for (uint32_t channel = 0; channel < num_output_; ++channel) {
        int32_t start_x = -padding_x_;
        for (uint32_t out_x = 0; out_x < out_width_; start_x += stride_x_,
          ++out_x) {
          int32_t start_y = -padding_y_;
          for (uint32_t out_y = 0; out_y < out_height_; start_y += stride_y_,
            ++out_y) {
            Dtype val = Dtype(0);
            for (uint32_t x = 0; x < filter_width_; ++x) {
              int32_t in_x = start_x + x;
              if (in_x < 0 || uint32_t(in_x) >= in_width) {
                continue;
              }
              for (uint32_t y = 0; y < filter_height_; ++y) {
                int32_t in_y = start_y + y;
                if (in_y < 0 || uint32_t(in_y) >= in_height) {
                  continue;
                }
                for (uint32_t in_channel = 0; in_channel < num_input;
                  ++in_channel) {
                  uint32_t filter_index =
                    ((channel * num_input + in_channel) * filter_height_ + y) *
                    filter_width_ + x;
                  uint32_t in_index =
                    in_offset + (in_channel * in_height + in_y) * in_width +
                    in_x;
                  val += in_data[in_index] * filter_data[filter_index];
                }
              }
            }
            uint32_t out_index =
              out_offset + (channel * out_height_ + out_y) * out_width_ +
              out_x;
            if (bias_data) {
              val += bias_data[channel];
            }
            out_data[out_index] = val;
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
    const Dtype* out_grad_data    = Parent::out_[0]->Grad();
    const Dtype* in_data          = Parent::in_[0]->Data();
    Dtype*       in_grad_data     = Parent::in_[0]->Grad();
    const Dtype* filter_data      = Parent::weight_[0]->Data();
    Dtype*       filter_grad_data = Parent::weight_[0]->Grad();
    Dtype*       bias_grad_data   = (Parent::weight_.size() > 1) ?
                                    Parent::weight_[1]->Grad() : nullptr;

    uint32_t in_width   = Parent::in_[0]->size[0];
    uint32_t in_height  = Parent::in_[0]->size[1];
    uint32_t num_input  = Parent::in_[0]->size[2];
    uint32_t batch_size = Parent::in_[0]->size[3];

    // in_grad     = filter * out_grad
    // filter_grad = in * out_grad
    // bias_grad   = out_grad
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t in_offset  = in_width   * in_height   * num_input   * batch;
      uint32_t out_offset = out_width_ * out_height_ * num_output_ * batch;
      for (uint32_t channel = 0; channel < num_output_; ++channel) {
        int32_t start_x = -padding_x_;
        for (uint32_t out_x = 0; out_x < out_width_; start_x += stride_x_,
          ++out_x) {
          int32_t start_y = -padding_y_;
          for (uint32_t out_y = 0; out_y < out_height_; start_y += stride_y_,
            ++out_y) {
            uint32_t grad_index = out_offset +
              (channel * out_height_ + out_y) * out_width_ + out_x;
            Dtype dv = out_grad_data[grad_index];
            for (uint32_t x = 0; x < filter_width_; ++x) {
              int32_t in_x = start_x + x;
              if (in_x < 0 || uint32_t(in_x) >= in_width) {
                continue;
              }
              for (uint32_t y = 0; y < filter_height_; ++y) {
                int32_t in_y = start_y + y;
                if (in_y < 0 || uint32_t(in_y) >= in_height) {
                  continue;
                }
                for (uint32_t in_channel = 0; in_channel < num_input;
                  ++in_channel) {
                  uint32_t filter_index =
                    ((channel * num_input + in_channel) * filter_height_ + y) *
                    filter_width_ + x;
                  uint32_t in_index =
                    in_offset + (in_channel * in_height + in_y) * in_width +
                    in_x;
                  in_grad_data[in_index] += filter_data[filter_index] * dv;
                  filter_grad_data[filter_index] += in_data[in_index] * dv;
                }
              }
            }
            if (bias_grad_data) {
              bias_grad_data[channel] += dv;
            }
          }
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_CONV_H_
