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


#ifndef CORE_LAYER_DROPOUT_H_
#define CORE_LAYER_DROPOUT_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <limits>
#include <random>
#include <vector>


namespace jik {


/*!
 *  \class  LayerDropout
 *  \brief  Dropout
 */
template <typename Dtype>
class LayerDropout: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Protected attributes
 protected:
  Dtype prob_;  // Probability to drop


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  LayerDropout(const char*                                     name,
               const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
               const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Probability to drop
    param.Get("prob", &prob_);

    // Create the mask and initialize it to 0
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);

    // Create 2 outputs, same size as the input
    // The first input is the result of the dropout, the second is the mask
    // There's no gradient for the mask as we don't try to learn it
    Parent::out_.resize(2);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size,
                                                   false);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerDropout() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    if (state.phase != State::PHASE_TRAIN) {
      // Dropout only during the training phase
      Parent::out_[0]->data = Parent::in_[0]->data;
      return;
    }

    // out = mask * in
    if (prob_ < std::numeric_limits<Dtype>::epsilon()) {
      // Nothing to drop: just copy the input to the output
      Parent::out_[0]->data = Parent::in_[0]->data;
      Parent::out_[1]->Set(Dtype(1));
    } else if (prob_ > Dtype(1) - std::numeric_limits<Dtype>::epsilon()) {
      // Drop everything: zero out the data
      Parent::out_[0]->Zero();
      Parent::out_[1]->Zero();
    } else {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<Dtype> dist(Dtype(0), Dtype(1));

      Dtype* mask_data = Parent::out_[1]->Data();
      Dtype scale = Dtype(1) / (Dtype(1) - prob_);
      for (uint32_t i = 0; i < Parent::out_[1]->Size(); ++i) {
        if (dist(gen) < prob_) {
          mask_data[i] = Dtype(0);
        } else {
          mask_data[i] = scale;
        }
      }

      Dtype*       out_data = Parent::out_[0]->Data();
      const Dtype* in_data  = Parent::in_[0]->Data();
      for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
        out_data[i] = mask_data[i] * in_data[i];
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
    const Dtype* mask_data     = Parent::out_[1]->Data();
    Dtype*       in_grad_data  = Parent::in_[0]->Grad();

    // in_grad = mask * out_grad
    for (uint32_t i = 0; i < Parent::out_[0]->Size(); ++i) {
      in_grad_data[i] += mask_data[i] * out_grad_data[i];
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_DROPOUT_H_
