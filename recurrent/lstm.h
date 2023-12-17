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


#ifndef RECURRENT_LSTM_H_
#define RECURRENT_LSTM_H_


#include <recurrent/recurrent.h>
#include <core/layer_add.h>
#include <core/layer_eltwise_mult.h>
#include <core/layer_mult.h>
#include <core/layer_tanh.h>
#include <core/layer_sigmoid.h>
#include <core/rand.h>
#include <memory>
#include <vector>


namespace jik {


/*!
 *  \class  Lstm
 *  \brief  Long Short Term Memory model
 */
template <typename Dtype>
class Lstm: public Recurrent<Dtype> {
  // Public types
 public:
  typedef Dtype            Type;
  typedef Recurrent<Dtype> Parent;


  // Protected attributes
 protected:
  std::vector<std::shared_ptr<Mat<Dtype>>> wix_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wih_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> bi_;           // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wfx_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wfh_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> bf_;           // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wox_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> woh_;          // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> bo_;           // Gates weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wcx_;          // Cell write weights
  std::vector<std::shared_ptr<Mat<Dtype>>> wch_;          // Cell write weights
  std::vector<std::shared_ptr<Mat<Dtype>>> bc_;           // Cell write weights
  std::shared_ptr<Mat<Dtype>>              whd_;          // Decoder weights
  std::shared_ptr<Mat<Dtype>>              bd_;           // Decoder weights
  std::shared_ptr<Mat<Dtype>>              wil_;          // Decoder weights
  std::vector<std::shared_ptr<Mat<Dtype>>> hidden_prev_;  // Previous hidden
  std::vector<std::shared_ptr<Mat<Dtype>>> cell_prev_;    // Previous cells
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
  Lstm(const char* name, uint32_t size_in,
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
      wix_.push_back(Rand<Dtype>::GenMat(hsize, size_prev, 1, batch_size,
                                         -hrange, hrange));
      wih_.push_back(Rand<Dtype>::GenMat(hsize, hsize, 1, batch_size,
                                         -hrange, hrange));
      bi_.push_back(std::make_shared<Mat<Dtype>>(hsize, 1, 1, batch_size));
      wfx_.push_back(Rand<Dtype>::GenMat(hsize, size_prev, 1, batch_size,
                                         -hrange, hrange));
      wfh_.push_back(Rand<Dtype>::GenMat(hsize, hsize, 1, batch_size,
                                         -hrange, hrange));
      bf_.push_back(std::make_shared<Mat<Dtype>>(hsize, 1, 1, batch_size));
      wox_.push_back(Rand<Dtype>::GenMat(hsize, size_prev, 1, batch_size,
                                         -hrange, hrange));
      woh_.push_back(Rand<Dtype>::GenMat(hsize, hsize, 1, batch_size,
                                         -hrange, hrange));
      bo_.push_back(std::make_shared<Mat<Dtype>>(hsize, 1, 1, batch_size));

      // Add the cell write weights
      wcx_.push_back(Rand<Dtype>::GenMat(hsize, size_prev, 1, batch_size,
                                         -hrange, hrange));
      wch_.push_back(Rand<Dtype>::GenMat(hsize, hsize, 1, batch_size,
                                         -hrange, hrange));
      bc_.push_back(std::make_shared<Mat<Dtype>>(hsize, 1, 1, batch_size));
    }

    // Create the decoder weights
    whd_ = Rand<Dtype>::GenMat(size_out, hsize, 1, batch_size,
                               -hrange, hrange);
    bd_  = std::shared_ptr<Mat<Dtype>>(std::make_shared<Mat<Dtype>>(
                                       size_out, 1, 1, batch_size));
    wil_ = Rand<Dtype>::GenMat(size_in, size_out, 1, batch_size,
                               -hrange, hrange);
  }

  /*!
   * Destructor.
   */
  virtual ~Lstm() {}

  /*!
   * Get the weights.
   *
   *  \param[out] weight: list of weights
   */
  virtual void GetWeight(std::vector<std::shared_ptr<Mat<Dtype>>>* weight) {
    weight->clear();
    weight->reserve(12 * hidden_size_.size() + 3);
    for (size_t i = 0; i < hidden_size_.size(); ++i) {
      weight->push_back(wix_[i]);
      weight->push_back(wih_[i]);
      weight->push_back(bi_[i]);
      weight->push_back(wfx_[i]);
      weight->push_back(wfh_[i]);
      weight->push_back(bf_[i]);
      weight->push_back(wox_[i]);
      weight->push_back(woh_[i]);
      weight->push_back(bo_[i]);
      weight->push_back(wcx_[i]);
      weight->push_back(wch_[i]);
      weight->push_back(bc_[i]);
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
      for (size_t d = 0; d < hidden_size_.size(); ++d) {
        hidden_prev_.push_back(std::make_shared<Mat<Dtype>>(
          hidden_size_[d], 1, 1, batch_size));
      }
    }
    if (cell_prev_.empty()) {
      for (size_t d = 0; d < hidden_size_.size(); ++d) {
        cell_prev_.push_back(std::make_shared<Mat<Dtype>>(hidden_size_[d],
                                                          1, 1, batch_size));
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
    std::vector<std::shared_ptr<Mat<Dtype>>> cell;
    for (size_t d = 0; d < hidden_size_.size(); ++d) {
      std::shared_ptr<Mat<Dtype>> in_vector;
      if (d == 0) {
        in_vector = x;
      } else {
        in_vector = hidden[d - 1];
      }

      std::shared_ptr<Mat<Dtype>> hidden_prev = hidden_prev_[d];
      std::shared_ptr<Mat<Dtype>> cell_prev   = cell_prev_[d];

      // Input gate
      std::shared_ptr<Mat<Dtype>> hi0 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wix_[d],
        in_vector}))[0];
      std::shared_ptr<Mat<Dtype>> hi1 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wih_[d],
        hidden_prev}))[0];
      std::shared_ptr<Mat<Dtype>> hi = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hi0, hi1}))[0];
      std::shared_ptr<Mat<Dtype>> biasi = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hi, bi_[d]}))[0];
      std::shared_ptr<Mat<Dtype>> gate_in = Parent::Add(
        std::make_shared<LayerSigmoid<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{biasi}))[0];

      // Forget gate
      std::shared_ptr<Mat<Dtype>> hf0 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wfx_[d],
        in_vector}))[0];
      std::shared_ptr<Mat<Dtype>> hf1 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wfh_[d],
        hidden_prev}))[0];
      std::shared_ptr<Mat<Dtype>> hf = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hf0, hf1}))[0];
      std::shared_ptr<Mat<Dtype>> biasf = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hf, bf_[d]}))[0];
      std::shared_ptr<Mat<Dtype>> gate_forget = Parent::Add(
        std::make_shared<LayerSigmoid<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{biasf}))[0];

      // Output gate
      std::shared_ptr<Mat<Dtype>> ho0 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wox_[d],
        in_vector}))[0];
      std::shared_ptr<Mat<Dtype>> ho1 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{woh_[d],
        hidden_prev}))[0];
      std::shared_ptr<Mat<Dtype>> ho = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{ho0, ho1}))[0];
      std::shared_ptr<Mat<Dtype>> biaso = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{ho, bo_[d]}))[0];
      std::shared_ptr<Mat<Dtype>> gate_out = Parent::Add(
        std::make_shared<LayerSigmoid<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{biaso}))[0];

      // Write operation on cells
      std::shared_ptr<Mat<Dtype>> hw0 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wcx_[d],
        in_vector}))[0];
      std::shared_ptr<Mat<Dtype>> hw1 = Parent::Add(
        std::make_shared<LayerMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{wch_[d],
        hidden_prev}))[0];
      std::shared_ptr<Mat<Dtype>> hw = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hw0, hw1}))[0];
      std::shared_ptr<Mat<Dtype>> biasw = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{hw, bc_[d]}))[0];
      std::shared_ptr<Mat<Dtype>> cell_write = Parent::Add(
        std::make_shared<LayerTanh<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{biasw}))[0];

      // Compute the new cell activation
      // Add what we want to keep from the cell
      std::shared_ptr<Mat<Dtype>> cell_retain = Parent::Add(
        std::make_shared<LayerEltwiseMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{gate_forget,
        cell_prev}))[0];
      // Add what we want to write to the cell
      std::shared_ptr<Mat<Dtype>> write_cell = Parent::Add(
        std::make_shared<LayerEltwiseMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{gate_in,
        cell_write}))[0];
      // Add the new cell content
      std::shared_ptr<Mat<Dtype>> cell_curr = Parent::Add(
        std::make_shared<LayerAdd<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{cell_retain,
        write_cell}))[0];

      // Compute the hidden state as a gated and saturated cell activations
      std::shared_ptr<Mat<Dtype>> tanhc = Parent::Add(
        std::make_shared<LayerTanh<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{cell_curr}))[0];
      std::shared_ptr<Mat<Dtype>> hidden_curr = Parent::Add(
        std::make_shared<LayerEltwiseMult<Dtype>>("",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{gate_out,
        tanhc}))[0];

      cell.push_back(cell_curr);
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
    cell_prev_   = cell;
  }

  /*!
   * Clear the previous iteration state.
   */
  virtual void ClearPrevState() {
    hidden_prev_.clear();
    cell_prev_.clear();
  }
};


}  // namespace jik


#endif  // RECURRENT_LSTM_H_
