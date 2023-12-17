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


#ifndef CORE_LAYER_H_
#define CORE_LAYER_H_


#include <core/mat.h>
#include <core/param.h>
#include <core/state.h>
#include <memory>
#include <vector>
#include <string>


namespace jik {


/*!
 *  \class  Layer
 *  \brief  Base layer class
 *
 * Any layer must derive from this base class and
 * implement the forward and backward functions.
 *
 * A layer is composed of input and output activations which are the
 * result of a given operation and calculated during the forward pass.
 * A layer eventually has weights that the model will try to learn.
 *
 * The gradients are calculated during the backward
 * pass and propagated back to the previous layer.
 *
 * The name layer here is used broadly for any function part of a network
 * taking N inputs and producing M outputs.
 * Some frameworks call this an op (operator) or unit while reserving the name
 * layer for a group of units with some weights
 * (e.g. Conv+ReLU+Pool = 1 layer with 3 units, or 1 layer of width 3).
 * We will call all these units layers, independently from the fact that they
 * may have some weight or not (i.e. if they learn some parameters or not).
 */
template <typename Dtype>
class Layer {
  // Public types
 public:
  typedef Dtype Type;


  // Protected attributes
 protected:
  std::string                              name_;     // Layer name
  std::vector<std::shared_ptr<Mat<Dtype>>> in_;       // Input  activations
  std::vector<std::shared_ptr<Mat<Dtype>>> out_;      // Output activations
  std::vector<std::shared_ptr<Mat<Dtype>>> weight_;   // Weights


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   */
  Layer(const char*                                     name,
        const std::vector<std::shared_ptr<Mat<Dtype>>>& in) {
    name_ = name;
    in_   = in;
  }

  /*!
   * Destructor.
   */
  virtual ~Layer() {}

  /*!
   * Get the layer name.
   *
   *  \return Layer name
   */
  const char* Name() const {
    return name_.c_str();
  }

  /*!
   * Get the layer input activations.
   *
   *  \return Input activations
   */
  const std::vector<std::shared_ptr<Mat<Dtype>>>& Input() const {
    return in_;
  }

  /*!
   * Get the layer output activations.
   *
   *  \return Output activations
   */
  const std::vector<std::shared_ptr<Mat<Dtype>>>& Output() const {
    return out_;
  }

  /*!
   * Set the batch size.
   *
   *  \param[in]  batch_size: batch size
   */
  void SetBatchSize(uint32_t batch_size) {
    for (size_t i = 0; i < in_.size(); ++i) {
      in_[i]->size[3] = batch_size;
    }
    for (size_t i = 0; i < out_.size(); ++i) {
      out_[i]->size[3] = batch_size;
    }
  }

  /*!
   * Clear the gradients.
   */
  virtual void ZeroGrad() {
    for (size_t i = 0; i < in_.size(); ++i) {
      in_[i]->ZeroGrad();
    }
    for (size_t i = 0; i < out_.size(); ++i) {
      out_[i]->ZeroGrad();
    }
    for (size_t i = 0; i < weight_.size(); ++i) {
      weight_[i]->ZeroGrad();
    }
  }

  /*!
   * Get the weights.
   *
   *  \param[out] weight: list of weights
   */
  void GetWeight(std::vector<std::shared_ptr<Mat<Dtype>>>* weight) const {
    weight->reserve(weight->size() + weight_.size());
    for (size_t i = 0; i < weight_.size(); ++i) {
      weight->push_back(weight_[i]);
    }
  }

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) = 0;

  /*!
   * Backward pass.
   * The backward pass calculates the inputs activations and weights
   * gradients in regard to the outputs activations gradients.
   *
   *  \param[in]  state: state
   */
  virtual void Backward(const State& state) = 0;
};


}  // namespace jik


#endif  // CORE_LAYER_H_
