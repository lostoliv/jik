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


#ifndef CORE_LAYER_ELTWISE_SCALE_H_
#define CORE_LAYER_ELTWISE_SCALE_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  EltwiseScaleLayer
 *  \brief  Eltwise scale layer
 */
template <typename Dtype>
class EltwiseScaleLayer: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Protected attributes
 protected:
  Dtype scale_;   // Scale
  Dtype bias_;    // Bias


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  EltwiseScaleLayer(const char*                                     name,
                    const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
                    const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Parameters
    param.Get("scale", Dtype(1), &scale_);
    param.Get("bias" , Dtype(0), &bias_);

    // Create 1 output, same size as the input
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
  }

  /*!
   * Destructor.
   */
  virtual ~EltwiseScaleLayer() {}

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

    // out = in * scale + bias
    for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
      out_data[i] = in_data[i] * scale_ + bias_;
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

    // in_grad = out_grad * scale
    for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
      in_grad_data[i] += out_grad_data[i] * scale_;
    }
  }
};

}  // namespace jik


#endif  // CORE_LAYER_ELTWISE_SCALE_H_
