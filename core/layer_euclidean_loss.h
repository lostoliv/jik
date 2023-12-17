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


#ifndef CORE_LAYER_EUCLIDEAN_LOSS_H_
#define CORE_LAYER_EUCLIDEAN_LOSS_H_


#include <core/layer_loss.h>
#include <memory>
#include <cmath>
#include <vector>


namespace jik {


/*!
 *  \class  LayerEuclideanLoss
 *  \brief  Euclidean (quadratic) loss function
 */
template <typename Dtype>
class LayerEuclideanLoss: public LayerLoss<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerLoss<Dtype>  Parent;


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name: layer name
   *  \param[in]  in  : input activations
   */
  LayerEuclideanLoss(const char*                                     name,
                     const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {
    Check(Parent::in_[0]->size[0] == Parent::in_[1]->size[0] &&
          Parent::in_[0]->size[1] == Parent::in_[1]->size[1] &&
          Parent::in_[0]->size[2] == Parent::in_[1]->size[2] &&
          Parent::in_[0]->size[3] == Parent::in_[1]->size[3],
          "Input layers must have the same size");

    // Create 1 more output, same size as the inputs
    // to save the result of the euclidean distance (square)
    // There's no gradient as we don't backpropagate it
    Parent::out_.resize(2);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size,
                                                   false);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerEuclideanLoss() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       loss_data = Parent::out_[0]->Data();
    Dtype*       out_data  = Parent::out_[1]->Data();
    const Dtype* in1_data  = Parent::in_[0]->Data();
    const Dtype* in2_data  = Parent::in_[1]->Data();

    loss_data[0] = Dtype(0);
    if (!Parent::in_[0]->Size()) {
      return;
    }

    // out0 = sum((in1 - in2) * (in1 - in2))
    // out1 = (in1 - in2) * (in1 - in2)
    Dtype inv_size = Dtype(1) / Parent::in_[0]->Size();
    for (uint32_t i = 0; i < Parent::in_[0]->Size(); ++i) {
      out_data[i]   = (in1_data[i] - in2_data[i]) *
                      (in1_data[i] - in2_data[i]);
      loss_data[0] += out_data[i] * inv_size;
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
    for (uint32_t i = 0; i < Parent::in_[0]->Size(); ++i) {
      Parent::in_[0]->grad->data[i] -= Parent::out_[1]->data[i];
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_EUCLIDEAN_LOSS_H_
