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


#ifndef CORE_LAYER_BATCH_NORM_H_
#define CORE_LAYER_BATCH_NORM_H_


#include <core/layer.h>
#include <core/log.h>
#include <memory>
#include <cmath>
#include <limits>
#include <vector>


namespace jik {


/*!
 *  \class  LayerBatchNorm
 *  \brief  Batch normalization
 */
template <typename Dtype>
class LayerBatchNorm: public Layer<Dtype> {
  // Public types
 public:
  typedef Dtype        Type;
  typedef Layer<Dtype> Parent;


  // Protected attributes
 protected:
  std::shared_ptr<Mat<Dtype>> mean_cur_;         // Current mean
  std::shared_ptr<Mat<Dtype>> std_dev_cur_;      // Current standard deviation
  Dtype                       moving_avg_;       // Moving average
  Dtype                       moving_avg_frac_;  // Moving average fraction


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  in   : input activations
   *  \param[in]  param: parameters
   */
  LayerBatchNorm(const char*                                     name,
                 const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
                 const Param&                                    param):
    Parent(name, in) {
    // Make sure we have 1 input
    Check(Parent::in_.size() == 1, "Layer '%s' must have 1 input",
          Parent::Name());

    // Moving average
    // The moving average is multiplied by the fraction after each step
    moving_avg_ = Dtype(1);
    param.Get("moving_avg_frac", Dtype(0.99), &moving_avg_frac_);

    // Create 2 weights: mean and standard deviation
    // The standard deviation is actually stored as its inverse for efficiency
    // We learn 1 mean and standard deviation per channel
    Parent::weight_.resize(2);
    Parent::weight_[0] = std::make_shared<Mat<Dtype>>(
      1, 1, Parent::in_[0]->size[2]);
    Parent::weight_[1] = std::make_shared<Mat<Dtype>>(
      1, 1, Parent::in_[0]->size[2]);

    // Temporary matrices for the current mean and standard deviation values
    // No gradient needed
    mean_cur_    = std::make_shared<Mat<Dtype>>(1, 1, Parent::in_[0]->size[2],
                                                1, false);
    std_dev_cur_ = std::make_shared<Mat<Dtype>>(1, 1, Parent::in_[0]->size[2],
                                                1, false);

    // Create 1 output, same size as the input
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(Parent::in_[0]->size);
  }

  /*!
   * Destructor.
   */
  virtual ~LayerBatchNorm() {}

  /*!
   * Calculate the mean and variance of a dataset.
   *
   *  \param[in]  data     : dataset
   *  \param[in]  data_size: dataset size
   *
   *  \param[out] mean     : mean value
   *  \param[out] variance : variance value
   */
  static void MeanVariance(const Dtype* data, uint32_t data_size,
                           Dtype* mean, Dtype* variance) {
    if (!data || !data_size) {
      if (mean) {
        *mean = Dtype(0);
      }
      if (variance) {
        *variance = Dtype(0);
      }
      return;
    }

    // Calculate the mean
    Dtype mean_val = Dtype(0);
    for (uint32_t i = 0; i < data_size; ++i) {
      mean_val += data[i];
    }
    mean_val /= data_size;
    if (mean) {
      *mean = mean_val;
    }

    if (variance) {
      // Calculate the variance
      Dtype variance_val = Dtype(0);
      for (uint32_t i = 0; i < data_size; ++i) {
        Dtype dx      = data[i] - mean_val;
        variance_val += dx * dx;
      }
      variance_val /= data_size;
      *variance = variance_val;
    }
  }

  /*!
   * Forward pass.
   * The forward pass calculates the outputs activations
   * in regard to the inputs activations and weights.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    Dtype*       out_data         = Parent::out_[0]->Data();
    const Dtype* in_data          = Parent::in_[0]->Data();
    Dtype*       mean_data        = Parent::weight_[0]->Data();
    Dtype*       std_dev_data     = Parent::weight_[1]->Data();
    Dtype*       mean_cur_data    = mean_cur_->Data();
    Dtype*       std_dev_cur_data = std_dev_cur_->Data();

    uint32_t data_size   = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];
    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];
    if (!batch_size) {
      return;
    }
    Dtype inv_batch_size = Dtype(1) / batch_size;

    if (state.phase == State::PHASE_TRAIN) {
      // Calculate the mean and variance for each channel across all batches
      // We only do this during the training phase
      // During testing, we only use the precomputed
      // global mean and standard deviation
      for (uint32_t channel = 0; channel < num_channel; ++channel) {
        mean_cur_data[channel]    = Dtype(0);
        std_dev_cur_data[channel] = Dtype(0);
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
          uint32_t offset = (batch * num_channel + channel) * data_size;
          Dtype mean_val, variance_val;
          MeanVariance(in_data + offset, data_size, &mean_val, &variance_val);
          mean_cur_data[channel]    += mean_val     * inv_batch_size;
          std_dev_cur_data[channel] += variance_val * inv_batch_size;
        }

        // Calculate the standard deviation from the variance
        // We actually save the inverse of the standard deviation
        // sqrt(var(in) + eps)
        std_dev_cur_data[channel] =
          Dtype(1) / std::sqrt(std_dev_cur_data[channel] +
          std::numeric_limits<Dtype>::epsilon());

        // Global mean and standard deviation
        mean_data[channel] = (Dtype(1) - moving_avg_) * mean_data[channel] +
                             moving_avg_ * mean_cur_data[channel];
        std_dev_data[channel] =
          (Dtype(1) - moving_avg_) * std_dev_data[channel] +
          moving_avg_ * std_dev_cur_data[channel];
      }

      // Update the moving average
      moving_avg_ *= moving_avg_frac_;
    }

    // Normalize each value with the mean and variance
    // out = (in - mean) / sqrt(var(in) + eps)
    for (uint32_t channel = 0; channel < num_channel; ++channel) {
      for (uint32_t batch = 0; batch < batch_size; ++batch) {
        uint32_t offset = (batch * num_channel + channel) * data_size;
        for (uint32_t i = 0; i < data_size; ++i) {
          out_data[offset + i] = (in_data[offset + i] - mean_data[channel]) *
                                 std_dev_data[channel];
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
    const Dtype* out_data      = Parent::out_[0]->Data();
    const Dtype* out_grad_data = Parent::out_[0]->Grad();
    Dtype*       in_grad_data  = Parent::in_[0]->Grad();
    const Dtype* std_dev_data  = Parent::weight_[1]->Data();

    uint32_t data_size   = Parent::out_[0]->size[0] * Parent::out_[0]->size[1];
    uint32_t num_channel = Parent::out_[0]->size[2];
    uint32_t batch_size  = Parent::out_[0]->size[3];

    // Temporary array to save "out_grad . out" element-wise product
    std::vector<Dtype> out_grad_data_dot_out_data(data_size);

    // in_grad = (out_grad - mean(out_grad) -
    //            mean(out_grad . out) . out) / sqrt(var(in) + eps)
    for (uint32_t channel = 0; channel < num_channel; ++channel) {
      for (uint32_t batch = 0; batch < batch_size; ++batch) {
        uint32_t offset = (batch * num_channel + channel) * data_size;
        // mean(out_grad)
        Dtype mean_out_grad_data;
        MeanVariance(out_grad_data + offset, data_size,
                     &mean_out_grad_data, nullptr);
        // out_grad . out
        for (uint32_t i = 0; i < data_size; ++i) {
          out_grad_data_dot_out_data[i] = out_grad_data[offset + i] *
                                          out_data[offset + i];
        }
        // mean(out_grad . out)
        Dtype mean_out_grad_data_dot_out_data;
        MeanVariance(&out_grad_data_dot_out_data[0], data_size,
                     &mean_out_grad_data_dot_out_data, nullptr);
        // (out_grad - mean(out_grad) - mean(out_grad . out) . out) /
        // sqrt(var(in) + eps)
        // We re-use 1 / sqrt(var(in) + eps) calculated during the forward pass
        for (uint32_t i = 0; i < data_size; ++i) {
          in_grad_data[offset + i] +=
            (out_grad_data[offset + i] - mean_out_grad_data -
            mean_out_grad_data_dot_out_data * out_data[offset + i]) *
            std_dev_data[channel];
        }
      }
    }
  }
};


}  // namespace jik


#endif  // CORE_LAYER_BATCH_NORM_H_
