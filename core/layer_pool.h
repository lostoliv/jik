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


#ifndef CORE_LAYER_POOL_H_
#define CORE_LAYER_POOL_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  LayerPool
 *  \brief  Matrice pooling base class
 */
template <typename Dtype>
class LayerPool: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Protected attributes
 protected:
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
  LayerPool(const char*                                     name,
            const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
            const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Parameters
    param.Get("filter_width" , &filter_width_);
    param.Get("filter_height", &filter_height_);
    param.Get("padding_x"    , &padding_x_);
    param.Get("padding_y"    , &padding_y_);
    param.Get("stride_x"     , &stride_x_);
    param.Get("stride_y"     , &stride_y_);

    // Calculate the output width and height based on padding and stride
    out_width_ = (Parent::in_[0]->size[0] + 2 * padding_x_ - filter_width_) /
                 stride_x_ + 1;
    out_height_ = (Parent::in_[0]->size[1] + 2 * padding_y_ - filter_height_) /
                  stride_y_ + 1;

    // Create 1 output
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(out_width_, out_height_,
      Parent::in_[0]->size[2], Parent::in_[0]->size[3]);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerPool() {}
};


}  // namespace jik


#endif  // CORE_LAYER_POOL_H_
