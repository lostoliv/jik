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


#include <core/arg_parse.h>
#include <core/log.h>
#include <core/dataset.h>
#include <core/layer_data.h>
#include <core/layer_scale.h>
#include <core/layer_euclidean_loss.h>
#include <core/solver_sgd.h>
#include <core/solver_rmsprop.h>
#include <random>


namespace jik {


/*!
 *  \class  LinearRegressionDataLayer
 *  \brief  Linear regression data layer
 */
template <typename Dtype>
class LinearRegressionDataLayer: public LayerData<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerData<Dtype>  Parent;


  // Protected attributes
 protected:
  Dtype    scale_;                // Scale
  Dtype    min_;                  // Min input value
  Dtype    max_;                  // Max input value
  Dtype    noise_;                // Noise value
  uint32_t dataset_train_size_;   // Dataset size (training)
  uint32_t dataset_test_size_;    // Dataset size (testing)
  uint32_t dataset_train_index_;  // Dataset index (training)
  uint32_t dataset_test_index_;   // Dataset index (testing)


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  param: parameters
   */
  LinearRegressionDataLayer(const char* name, const Param& param):
    LayerData<Dtype>(name) {
    // Parameters
    std::string dataset_path;
    uint32_t batch_size;
    param.Get("batch_size", &batch_size);
    param.Get("scale"     , &scale_);
    param.Get("min"       , &min_);
    param.Get("noise"     , &noise_);
    param.Get("max"       , &max_);
    param.Get("size_train", &dataset_train_size_);
    param.Get("size_test" , &dataset_test_size_);

    Report(kInfo, "Generating %d train and %d test values between [%f, %f] "
           "and scaling them by %f (with a noise of %f)",
           dataset_train_size_, dataset_test_size_,
           min_, max_, scale_, noise_);

    // Set index at the beginning of the dataset
    dataset_train_index_ = dataset_test_index_ = 0;

    // Create 2 outputs: output data itself and
    // label data that we try to conform to
    // There's no gradient for the labels as we don't backpropagate them
    Parent::out_.resize(2);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(1, 1, 1, batch_size);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(1, 1, 1, batch_size, false);
  }

  /*!
   * Destructor.
   */
  virtual ~LinearRegressionDataLayer() {}

  /*!
   * Get the test index.
   *
   *  \return Test index
   */
  uint32_t TestIndex() const {
    return dataset_test_index_;
  }

  /*!
   * Check if testing is done
   * (i.e. if we are at the end of the testing dataset).
   *
   *  \return Testing done?
   */
  bool TestingDone() {
    // We are done if we are at the end of the testing dataset
    bool testing_done = dataset_test_index_ >= dataset_test_size_;

    if (testing_done) {
      // If we are done, we rewind
      dataset_test_index_ = 0;
    }

    return testing_done;
  }

  /*!
   * Forward pass.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    // Get the proper dataset (either training or testing one)
    uint32_t* dataset_index, *dataset_size;
    if (state.phase == State::PHASE_TRAIN) {
      dataset_index = &dataset_train_index_;
      dataset_size  = &dataset_train_size_;
    } else {
      dataset_index = &dataset_test_index_;
      dataset_size  = &dataset_test_size_;
    }

    if (!*dataset_size) {
      Report(kError, "Empty dataset");
      Parent::out_[0]->Zero();
      Parent::out_[1]->Zero();
      return;
    }

    if (*dataset_index >= *dataset_size) {
      Report(kError, "Invalid dataset index");
      Parent::out_[0]->Zero();
      Parent::out_[1]->Zero();
      return;
    }

    Dtype* output_data = Parent::out_[0]->Data();
    Dtype* label_data  = Parent::out_[1]->Data();

    uint32_t batch_size = Parent::out_[0]->size[3];

    bool testing_done = false;

    std::mt19937 gen;
    std::uniform_real_distribution<Dtype> dist(min_, max_);
    Dtype minmax = scale_ * (max_ - min_);
    std::uniform_real_distribution<Dtype> dist_noise(Dtype(0), minmax);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      // Generate a unique seed based on the data index
      uint32_t seed = *dataset_index;
      if (state.phase == State::PHASE_TEST) {
        // Make sure that training and testing data is different
        seed += dataset_train_size_;
      }
      gen.seed(seed);
      // Input value
      output_data[batch] = dist(gen);
      // Output value: N * input
      // We will try to find N through linear regression
      label_data[batch] = scale_ * output_data[batch];
      if (std::abs(noise_) > std::numeric_limits<Dtype>::epsilon()) {
        // We add some noise to make sure the output is not a linear
        // combination of the input to make it more difficult to learn
        label_data[batch] += noise_ * (dist_noise(gen) - Dtype(0.5) * minmax);
      }

      // Go to the next value
      if (++*dataset_index >= *dataset_size) {
        if (state.phase == State::PHASE_TRAIN) {
          // Rewind
          *dataset_index = 0;
        } else {
          // Clamp
          *dataset_index = *dataset_size - 1;
          testing_done   = true;
        }
      }
    }

    if (testing_done) {
      // Mark the testing dataset as done
      *dataset_index = *dataset_size;
    }
  }
};


/*!
 *  \class  LinearRegressionModel
 *  \brief  LinearRegression model
 */
template <typename Dtype>
class LinearRegressionModel: public Model<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Model<Dtype>  Parent;


  // Protected attributes
 protected:
  Dtype                              scale_;        // Scale
  std::shared_ptr<Mat<Dtype>>        label_;        // Labels
  std::shared_ptr<Mat<Dtype>>        dist2_;        // Euclidean distances
  std::shared_ptr<LayerScale<Dtype>> scale_layer_;  // Scale layer


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name      : model name
   *  \param[in]  batch_size: matrix size (batch size)
   *  \param[in]  scale     : scale value to learn
   *  \param[in]  min       : data min value (lower boundary)
   *  \param[in]  max       : data max value (upper boundary)
   *  \param[in]  noise     : noise intensity to add to the data output
   *  \param[in]  size_train: training set size
   *  \param[in]  size_test : testing set size
   */
  LinearRegressionModel(const char* name, uint32_t batch_size,
                        Dtype scale, Dtype min, Dtype max, Dtype noise,
                        uint32_t size_train, uint32_t size_test):
    Model<Dtype>(name) {
    // Save the scale
    scale_ = scale;

    // Input layer parameters
    Param data_param;
    data_param.Add("batch_size", batch_size);
    data_param.Add("scale"     , scale);
    data_param.Add("min"       , min);
    data_param.Add("max"       , max);
    data_param.Add("noise"     , noise);
    data_param.Add("size_train", size_train);
    data_param.Add("size_test" , size_test);

    // Input layer
    std::vector<std::shared_ptr<Mat<Dtype>>> out = Parent::Add(
      std::make_shared<LinearRegressionDataLayer<Dtype>>("data1", data_param));

    // Model input (images) and labels
    Parent::in_ = out[0];
    label_      = out[1];

    // Output of previous layer
    Parent::out_ = Parent::in_;

    // Scale layer (we disable the bias term, we only learn the scale)
    Param scale_param;
    scale_param.Add("use_bias", false);
    scale_layer_ = std::make_shared<LayerScale<Dtype>>("scale",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      scale_param);
    Parent::out_ = Parent::Add(scale_layer_)[0];

    // Loss function (euclidean distance)
    out = Parent::Add(std::make_shared<LayerEuclideanLoss<Dtype>>("loss",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_,
      label_}));
    Parent::out_ = out[0];
    dist2_       = out[1];
  }

  /*!
   * Destructor.
   */
  virtual ~LinearRegressionModel() {}

  /*!
   * Get the learnt scale.
   *
   *  \return Learnt scale
   */
  Dtype ScaleLearnt() const {
    std::vector<std::shared_ptr<Mat<Dtype>>> weight;
    scale_layer_->GetWeight(&weight);
    if (weight.size() != 1 && weight[0]->Size() != 1) {
      Report(kError, "There should be exactly 1 weight in layer '%s' "
             "(the scale learnt by model '%s')",
             scale_layer_->Name(), Parent::Name());
      return Dtype(0);
    }
    return weight[0]->Data()[0];
  }

  /*!
   * Graph testing (inference).
   *
   *  \return     Accuracy
   */
  virtual Dtype Test() {
    // Create the state
    State state(State::PHASE_TEST);

    // Get the data layer to keep track of the testing index
    std::shared_ptr<LinearRegressionDataLayer<Dtype>> linear_regression_data =
      std::dynamic_pointer_cast<LinearRegressionDataLayer<Dtype>>(
      Parent::DataLayer());
    if (!linear_regression_data) {
      Report(kError, "No data layer found in model '%s'", Parent::Name());
      return Dtype(0);
    }

    uint32_t step = 0;
    Dtype acc     = Dtype(0);
    while (!linear_regression_data->TestingDone()) {
      // Current test index
      uint32_t index = linear_regression_data->TestIndex();

      // Inference
      Parent::Forward(state);

      // Actual batch size = where we are - where we were
      uint32_t actual_batch_size = linear_regression_data->TestIndex() - index;

      const Dtype* label_data = label_->Data();
      const Dtype* dist2_data = dist2_->Data();

      // Batch accuracy
      Dtype bacc = Dtype(0);
      for (uint32_t batch = 0; batch < actual_batch_size; ++batch) {
        Dtype dist2_orig = scale_ * label_data[batch] - label_data[batch];
        dist2_orig      *= dist2_orig;
        bacc            += Dtype(1) - dist2_data[batch] / dist2_orig;
      }

      // Go to next step and accumulate the accuracy
      ++step;
      acc += bacc / actual_batch_size;
    }

    // Overall accuracy
    if (step) {
      return acc / step;
    }
    return Dtype(1);
  }
};


}  // namespace jik


int main(int argc, char* argv[]) {
  using namespace jik;  // NOLINT(build/namespaces)

  // 32-bit float quantization
  typedef float Dtype;

  // Hyperparameters
  ArgParse arg(argc, argv);
  const char* model_path   = arg.Arg("-model");
  const char* model_name   = arg.Arg("-name");
  const char* solver_type  = arg.Arg("-solver");
  bool        train        = arg.ArgExists("-train");
  uint32_t batch_size;
  Dtype learning_rate, decay_rate, momentum,
        reg, clip, lr_scale, mult, min, max, noise;
  uint32_t num_step, print_each, test_each, save_each, lr_scale_each;
  arg.Arg<uint32_t>("-batchsize"  , 1                   , &batch_size);
  arg.Arg<Dtype>   ("-lr"         , Dtype(0.01)         , &learning_rate);
  arg.Arg<Dtype>   ("-decayrate"  , Dtype(0.999)        , &decay_rate);
  arg.Arg<Dtype>   ("-momentum"   , Dtype(0.9)          , &momentum);
  arg.Arg<Dtype>   ("-reg"        , Dtype(0.001)        , &reg);
  arg.Arg<Dtype>   ("-clip"       , Dtype(5)            , &clip);
  arg.Arg<uint32_t>("-numstep"    , 10000               , &num_step);
  arg.Arg<uint32_t>("-printeach"  , 0                   , &print_each);
  arg.Arg<uint32_t>("-testeach"   , 0                   , &test_each);
  arg.Arg<uint32_t>("-saveeach"   , 0                   , &save_each);
  arg.Arg<uint32_t>("-lrscaleeach", 0                   , &lr_scale_each);
  arg.Arg<Dtype>   ("-lrscale"    , Dtype(1)            , &lr_scale);
  arg.Arg<Dtype>   ("-scale"      , Dtype(3.14159265359), &mult);
  arg.Arg<Dtype>   ("-min"        , Dtype(0)            , &min);
  arg.Arg<Dtype>   ("-max"        , Dtype(1)            , &max);
  arg.Arg<Dtype>   ("-noise"      , Dtype(0.001)        , &noise);

  if ((!train && !model_path) || arg.ArgExists("-h")) {
    Report(kInfo, "Usage: %s [-train] [-scale <SCALE>] [-min <MIN>] "
           "[-max <MAX>] [-noise <NOISE>] "
           "[-model <path/to/linear_regression/model>]", argv[0]);
    return -1;
  }

  // Default model and solver names
  if (!model_name) {
    model_name = "linear_regression";
  }
  if (!solver_type) {
    solver_type = "rmsprop";
  }

  // Printing hyperparameters
  Report(kInfo, "Batch size              : %d", batch_size);
  Report(kInfo, "Learning rate           : %f", learning_rate);
  Report(kInfo, "Decay rate              : %f", decay_rate);
  Report(kInfo, "Momentum                : %f", momentum);
  Report(kInfo, "L2 regularization       : %f", reg);
  Report(kInfo, "Gradient clipping       : %f", clip);
  Report(kInfo, "Number of steps         : %d", num_step);
  Report(kInfo, "Print each              : %d", print_each);
  Report(kInfo, "Test each               : %d", test_each);
  Report(kInfo, "Save each               : %d", save_each);
  Report(kInfo, "Scale learning rate each: %d", lr_scale_each);
  Report(kInfo, "Learning rate scale     : %f", lr_scale);

  // Create the model: make sure we have enough data to cover exactly 1 epoch
  LinearRegressionModel<Dtype> model(model_name, batch_size,
                                     mult, min, max, noise,
                                     num_step * batch_size,
                                     num_step * batch_size);

  // Load the model if one is specified
  if (model_path) {
    size_t size = model.Load(model_path);
    Report(kInfo, "Loading model '%s' (%ld byte(s))", model_path, size);
  }

  // Testing the model only
  if (!train) {
    Report(kInfo, "Testing model '%s'", model_name);
    Dtype acc = model.Test();
    Report(kInfo, "Accuracy: %f", acc);
    return 0;
  }

  Solver<Dtype>* solver;
  if (!std::strcmp(solver_type, "sgd")) {
    Report(kInfo, "Creating SGD solver");
    solver = new SolverSGD<Dtype>(print_each, test_each, save_each,
                                  lr_scale_each, lr_scale,
                                  momentum, reg, clip);
  } else if (!std::strcmp(solver_type, "rmsprop")) {
    Report(kInfo, "Creating RMSprop solver");
    solver = new SolverRMSprop<Dtype>(print_each, test_each, save_each,
                                      lr_scale_each, lr_scale, decay_rate,
                                      reg, clip);
  } else {
    Report(kError, "Unknown solver type '%s'", solver_type);
    return -1;
  }

  // Train the model
  if (!solver->Train(&model, num_step, learning_rate)) {
    return -1;
  }

  // Clean
  delete solver;

  Report(kInfo, "Scale learnt by model '%s': %f (loss %f)",
         model.Name(), model.ScaleLearnt(), model.Loss());

  return 0;
}
