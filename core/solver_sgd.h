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


#ifndef CORE_SOLVER_SGD_H_
#define CORE_SOLVER_SGD_H_


#include <core/solver.h>
#include <memory>


namespace jik {


/*!
 *  \class  SolverSGD
 *  \brief  Stochastic gradient descent solver
 */
template <typename Dtype>
class SolverSGD: public Solver<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Solver<Dtype> Parent;


  // Protected attributes
 protected:
  Dtype momentum_;  // Momentum
  Dtype reg_;       // L2 regularization
  Dtype clip_;      // Gradient clipping value


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  print_each   : print the model stats every n steps
   *  \param[in]  test_each    : test the model every n steps
   *  \param[in]  save_each    : save the model every n steps
   *  \param[in]  lr_scale_each: save the model every n steps
   *  \param[in]  lr_scale     : learning rate scale
   *  \param[in]  momentum     : momentum
   *  \param[in]  reg          : L2 regularization
   *  \param[in]  clip         : gradient clipping
   */
  SolverSGD(uint32_t print_each, uint32_t test_each, uint32_t save_each,
            uint32_t lr_scale_each, Dtype lr_scale,
            Dtype momentum, Dtype reg, Dtype clip):
    Parent(print_each, test_each, save_each, lr_scale_each, lr_scale) {
    momentum_ = momentum;
    reg_      = reg;
    clip_     = clip;
  }

  /*!
   * Destructor.
   */
  virtual ~SolverSGD() {}

  /*!
   * Stochastic gradient descent.
   *
   *  \param[in]  weight       : weights
   *  \param[in]  weight_prev  : previous weights
   *  \param[in]  batch_size   : batch size
   *  \param[in]  learning_rate: learning rate
   *  \param[in]  momentum     : momentum
   *  \param[in]  reg          : L2 regularization
   *  \param[in]  clip         : gradient clipping
   */
  static void SGD(const std::shared_ptr<Mat<Dtype>>& weight,
                  const std::shared_ptr<Mat<Dtype>>& weight_prev,
                  uint32_t batch_size, Dtype learning_rate,
                  Dtype momentum, Dtype reg, Dtype clip) {
    Dtype*       weight_data      = weight->Data();
    const Dtype* weight_grad_data = weight->Grad();
    Dtype*       weight_prev_data = weight_prev->Data();

    for (uint32_t i = 0; i < weight->Size(); ++i) {
      // SGD with momentum
      Dtype dv = weight_grad_data[i] / batch_size;
      dv       = momentum * weight_prev_data[i] + dv;

      // Save previous value for next iteration
      weight_prev_data[i] = dv;

      // Gradient clip
      if (dv > clip) {
        dv = clip;
      } else if (dv < -clip) {
        dv = -clip;
      }

      // Update and regularize
      weight_data[i] -= learning_rate * (dv + reg * weight_data[i]);
    }
  }

  /*!
   * Learning function.
   *
   *  \param[in]  batch_size   : batch size
   *  \param[in]  learning_rate: learning rate
   */
  virtual void Learn(uint32_t batch_size, Dtype learning_rate) const {
    for (size_t i = 0; i < Parent::weight_.size(); ++i) {
      SGD(Parent::weight_[i], Parent::weight_prev_[i],
          batch_size, learning_rate, momentum_, reg_, clip_);
    }
  }
};


}  // namespace jik


#endif  // CORE_SOLVER_SGD_H_
