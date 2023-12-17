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


#ifndef CORE_LAYER_RELU_H_
#define CORE_LAYER_RELU_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <algorithm>
#include <vector>


namespace jik {


/*!
 *  \class  LayerRelu
 *  \brief  RELU (Rectified Linear Unit) activation function
 */
template <typename Dtype>
class LayerRelu: public Layer<Dtype> {
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
  LayerRelu(const char*                                     name,
            const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Create 1 output, same size as the input
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerRelu() {}

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

    // RELU activation is thresholded at zero
    // out = in if in > 0, 0 otherwise
    for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
      out_data[i] = std::max(Dtype(0), in_data[i]);
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
    const Dtype* out_data      = Parent::out_[0]->Data();
    const Dtype* out_grad_data = Parent::out_[0]->Grad();
    Dtype*       in_grad_data  = Parent::in_[0]->Grad();

    // in_grad = out_grad if out > 0, 0 otherwise
    for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
      if (out_data[i] > Dtype(0)) {
        in_grad_data[i] += out_grad_data[i];
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_RELU_H_
