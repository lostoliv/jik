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


#ifndef CORE_LAYER_SOFTMAX_LOSS_H_
#define CORE_LAYER_SOFTMAX_LOSS_H_


#include <core/layer_loss.h>
#include <memory>
#include <cmath>
#include <vector>


namespace jik {


/*!
 *  \class  LayerSoftMaxLoss
 *  \brief  Softmax + multinomial logistic loss function
 */
template <typename Dtype>
class LayerSoftMaxLoss: public LayerLoss<Dtype> {
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
  LayerSoftMaxLoss(const char*                                     name,
                   const std::vector<std::shared_ptr<Mat<Dtype>>>& in):
    Parent(name, in) {
      Check(Parent::in_[0]->size[3] == Parent::in_[1]->size[3],
            "Layer '%s' inputs must have the same batch size", Parent::Name());
      Check(Parent::in_[1]->size[0] == 1 &&
            Parent::in_[1]->size[1] == 1 &&
            Parent::in_[1]->size[2] == 1,
            "Layer '%s' labels input must have a size 1x1x1xBatchSize");

    // Create 1 more output, same size as the inputs
    // to save the results of the softmax (probabilities)
    // There's no gradient as we don't backpropagate it
    Parent::out_.resize(2);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size,
                                                   false);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerSoftMaxLoss() {}

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       loss_data  = Parent::out_[0]->Data();
    Dtype*       out_data   = Parent::out_[1]->Data();
    const Dtype* in_data    = Parent::in_[0]->Data();
    const Dtype* label_data = Parent::in_[1]->Data();

    loss_data[0] = Dtype(0);

    uint32_t data_size = Parent::out_[1]->size[0] * Parent::out_[1]->size[1] *
                         Parent::out_[1]->size[2];
    uint32_t batch_size = Parent::out_[1]->size[3];
    if (!data_size || !batch_size) {
      return;
    }

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      // Find the max value for the current data
      uint32_t offset = batch * data_size;
      Dtype val_max   = in_data[offset];
      for (uint32_t i = 1; i < data_size; ++i) {
        if (in_data[offset + i] > val_max) {
          val_max = in_data[offset + i];
        }
      }

      // out = norm(exp(in - max))
      Dtype sum = Dtype(0);
      for (uint32_t i = 0; i < data_size; ++i) {
        out_data[offset + i] = std::exp(in_data[offset + i] - val_max);
        sum                 += out_data[offset + i];
      }
      sum = Dtype(1) / sum;
      for (uint32_t i = 0; i < data_size; ++i) {
        out_data[offset + i] *= sum;
      }
    }

    // Cross entropy between the prediction (output of the network)
    // and the label (true probability)
    Dtype inv_size = Dtype(1) / batch_size;
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t index = batch * data_size + uint32_t(label_data[batch]);
      loss_data[0] -= std::log(out_data[index]) * inv_size;
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
    Dtype*       in_grad_data = Parent::in_[0]->Grad();
    const Dtype* label_data   = Parent::in_[1]->Data();

    uint32_t data_size  = Parent::out_[1]->size[0] * Parent::out_[1]->size[1] *
                          Parent::out_[1]->size[2];
    uint32_t batch_size = Parent::out_[1]->size[3];

    Parent::in_[0]->grad->data = Parent::out_[1]->data;
    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      uint32_t index       = batch * data_size + uint32_t(label_data[batch]);
      in_grad_data[index] -= Dtype(1);
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_SOFTMAX_LOSS_H_
