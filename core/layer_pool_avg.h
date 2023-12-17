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


#ifndef CORE_LAYER_POOL_AVG_H_
#define CORE_LAYER_POOL_AVG_H_


#include <core/layer_pool.h>
#include <memory>
#include <limits>
#include <vector>


namespace jik {


/*!
 *  \class  LayerPoolAvg
 *  \brief  Matrice average pooling
 */
template <typename Dtype>
class LayerPoolAvg: public LayerPool<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerPool<Dtype>  Parent;


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  LayerPoolAvg(const char*                                     name,
               const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
               const Param&                                    param):
    Parent(name, in, param) {}

  /*!
   * Destructor.
   */
  virtual ~LayerPoolAvg() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data = Parent::out_[0]->Data();
    const Dtype* in_data  = Parent::in_[0]->Data();

    uint32_t in_width    = Parent::in_[0]->size[0];
    uint32_t in_height   = Parent::in_[0]->size[1];
    uint32_t num_channel = Parent::in_[0]->size[2];
    uint32_t batch_size  = Parent::in_[0]->size[3];

    // out = ave(in, kernel_x, kernel_y)
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t in_offset  = in_width * in_height * num_channel * batch;
      uint32_t out_offset = Parent::out_width_ * Parent::out_height_ *
                            num_channel * batch;
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        int32_t start_x = -Parent::padding_x_;
        for (uint32_t out_x = 0; out_x < Parent::out_width_;
          start_x += Parent::stride_x_, ++out_x) {
          int32_t start_y = -Parent::padding_y_;
          for (uint32_t out_y = 0; out_y < Parent::out_height_;
            start_y += Parent::stride_y_, ++out_y) {
            Dtype val      = Dtype(0);
            uint32_t count = 0;
            for (uint32_t x = 0; x < Parent::filter_width_; ++x) {
              int32_t in_x = start_x + x;
              if (in_x < 0 || uint32_t(in_x) >= in_width) {
                continue;
              }
              for (uint32_t y = 0; y < Parent::filter_height_; ++y) {
                int32_t in_y = start_y + y;
                if (in_y < 0 || uint32_t(in_y) >= in_height) {
                  continue;
                }
                uint32_t in_index =
                  in_offset + (channel * in_height + in_y) * in_width + in_x;
                val += in_data[in_index];
                ++count;
              }
            }
            uint32_t out_index =
              out_offset + (channel * Parent::out_height_ + out_y) *
              Parent::out_width_ + out_x;
            out_data[out_index] = val / count;
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
    Dtype*       in_grad_data  = Parent::in_[0]->Grad();

    uint32_t in_width    = Parent::in_[0]->size[0];
    uint32_t in_height   = Parent::in_[0]->size[1];
    uint32_t num_channel = Parent::in_[0]->size[2];
    uint32_t batch_size  = Parent::in_[0]->size[3];

    // in_grad = out_grad
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t in_offset  = in_width * in_height * num_channel * batch;
      uint32_t out_offset = Parent::out_width_ * Parent::out_height_ *
                            num_channel * batch;
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        int32_t start_x = -Parent::padding_x_;
        for (uint32_t out_x = 0; out_x < Parent::out_width_;
          start_x += Parent::stride_x_, ++out_x) {
          int32_t start_y = -Parent::padding_y_;
          for (uint32_t out_y = 0; out_y < Parent::out_height_;
            start_y += Parent::stride_y_, ++out_y) {
            uint32_t out_index =
              out_offset + (channel * Parent::out_height_ + out_y) *
              Parent::out_width_ + out_x;
            for (uint32_t x = 0; x < Parent::filter_width_; ++x) {
              int32_t in_x = start_x + x;
              if (in_x < 0 || uint32_t(in_x) >= in_width) {
                continue;
              }
              for (uint32_t y = 0; y < Parent::filter_height_; ++y) {
                int32_t in_y = start_y + y;
                if (in_y < 0 || uint32_t(in_y) >= in_height) {
                  continue;
                }
                uint32_t in_index =
                  in_offset + (channel * in_height + in_y) * in_width + in_x;
                in_grad_data[in_index] += out_grad_data[out_index];
              }
            }
          }
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_POOL_AVG_H_
