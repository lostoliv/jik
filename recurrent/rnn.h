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


#ifndef RECURRENT_RNN_H_
#define RECURRENT_RNN_H_


#include <recurrent/recurrent.h>
#include <core/rand.h>
#include <core/layer_add.h>
#include <core/layer_mult.h>
#include <core/layer_relu.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  Rnn
 *  \brief  Recurrent Neural Network model
 */
template <typename Dtype>
class Rnn: public Recurrent<Dtype> {
  // Public types
 public:
  typedef Dtype            Type;
  typedef Recurrent<Dtype> Parent;


  // Protected attributes
 protected:
  std::vector<std::shared_ptr<Mat<Dtype>>> wxh_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> whh_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> bhh_;          // Gates weights
  std::shared_ptr<Mat<Dtype>>              whd_;          // Decoder weights
  std::shared_ptr<Mat<Dtype>>              bd_;           // Decoder weights
  std::shared_ptr<Mat<Dtype>>              wil_;          // Decoder weights
  std::vector<std::shared_ptr<Mat<Dtype>>> hidden_prev_;  // Previous hidden
  std::vector<uint32_t>                    hidden_size_;  // Hidden state size


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name       : model name
   *  \param[in]  size_in    : input size
   *  \param[in]  hidden_size: hidden state size
   *  \param[in]  size_out   : output size
   *  \param[in]  range      : value range ([-range/2, range/2])
   *  \param[in]  batch_size : batch size
   */
  Rnn(const char* name, uint32_t size_in,
      const std::vector<uint32_t>& hidden_size,
      uint32_t size_out, Dtype range,
      uint32_t batch_size): Recurrent<Dtype>(name) {
    hidden_size_ = hidden_size;

    Dtype hrange   = 0.5f * range;
    uint32_t hsize = 0;
    for (size_t d = 0; d < hidden_size_.size(); ++d) {
      uint32_t size_prev;
      if (d == 0) {
        size_prev = size_in;
      } else {
        size_prev = hidden_size_[d - 1];
      }
      hsize = hidden_size_[d];

      // Add the gates weights
      wxh_.push_back(Rand<Dtype>::GenMat(hsize, size_prev, 1, batch_size,
                                         -hrange, hrange));
      whh_.push_back(Rand<Dtype>::GenMat(hsize, hsize, 1, batch_size,
                                         -hrange, hrange));
      bhh_.push_back(std::make_shared<Mat<Dtype>>(hsize, 1, 1, batch_size));
    }

    // Create the decoder weights
    whd_ = Rand<Dtype>::GenMat(size_out, hsize, 1, batch_size,
                               -hrange, hrange);
    bd_  = std::make_shared<Mat<Dtype>>(size_out, 1, 1, batch_size);
    wil_ = Rand<Dtype>::GenMat(size_in, size_out, 1, batch_size,
                               -hrange, hrange);
  }

  /*!
   * Destructor.
   */
  virtual ~Rnn() {}

  /*!
   * Get the weights.
   *
   *  \param[out] weight: list of weights
   */
  virtual void GetWeight(std::vector<std::shared_ptr<Mat<Dtype>>>* weight) {
    weight->clear();
    weight->reserve(3 * hidden_size_.size() + 3);
    for (size_t i = 0; i < hidden_size_.size(); ++i) {
      weight->push_back(wxh_[i]);
      weight->push_back(whh_[i]);
      weight->push_back(bhh_[i]);
    }
    weight->push_back(whd_);
    weight->push_back(bd_);
    weight->push_back(wil_);
  }

  /*!
   * Create at a specific index.
   *
   *  \param[in]  index: data index
   */
  virtual void Create(uint32_t index) {
    // Clear all the layers
    Parent::Clear();

    uint32_t batch_size = wil_->size[3];

    if (hidden_prev_.empty()) {
      for (uint32_t d = 0; d < hidden_size_.size(); ++d) {
        hidden_prev_.push_back(std::make_shared<Mat<Dtype>>(
          hidden_size_[d], 1, 1, batch_size));
      }
    }

    Parent::in_ = std::make_shared<Mat<Dtype>>(wil_->size[1],
                                               1, 1, batch_size);
    Parent::in_->Data()[index] = Dtype(1);

    std::shared_ptr<Mat<Dtype>> x = Parent::Add(
      std::make_shared<LayerMult<Dtype>>("",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wil_,
      Parent::in_}))[0];

    std::vector<std::shared_ptr<Mat<Dtype>>> hidden;
    for (size_t d = 0; d < hidden_size_.size(); ++d) {
      std::shared_ptr<Mat<Dtype>> in_vector;
      if (d == 0) {
        in_vector = x;
      } else {
        in_vector = hidden[d - 1];
      }
      std::shared_ptr<Mat<Dtype>>& hidden_prev = hidden_prev_[d];

      std::shared_ptr<Mat<Dtype>> h0 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wxh_[d],
        in_vector}))[0];
      std::shared_ptr<Mat<Dtype>> h1 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{whh_[d],
        hidden_prev}))[0];
      std::shared_ptr<Mat<Dtype>> h01 = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{h0, h1}))[0];
      std::shared_ptr<Mat<Dtype>> bias = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{h01, bhh_[d]}))[0];
      std::shared_ptr<Mat<Dtype>> hidden_curr = Parent::Add(
        std::make_shared<LayerRelu<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{bias}))[0];

      hidden.push_back(hidden_curr);
    }

    // Decoder
    std::shared_ptr<Mat<Dtype>> hd = Parent::Add(
      std::make_shared<LayerMult<Dtype>>("",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{whd_,
      hidden[hidden.size() - 1]}))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerAdd<Dtype>>("",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hd, bd_}))[0];

    hidden_prev_ = hidden;
  }

  /*!
   * Clear the previous iteration state.
   */
  virtual void ClearPrevState() {
    hidden_prev_.clear();
  }
};


}  // namespace jik


#endif  // RECURRENT_RNN_H_
