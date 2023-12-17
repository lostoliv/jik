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


#include <sys/stat.h>
#include <core/arg_parse.h>
#include <core/log.h>
#include <core/dataset.h>
#include <core/layer_data.h>
#include <core/layer_batch_norm.h>
#include <core/layer_scale.h>
#include <core/layer_conv.h>
#include <core/layer_relu.h>
#include <core/layer_pool_max.h>
#include <core/layer_inner_product.h>
#include <core/layer_softmax_loss.h>
#include <core/solver_sgd.h>
#include <core/solver_rmsprop.h>


namespace jik {


/*!
 *  \class  Cifar10Dataset
 *  \brief  Cifar10 dataset
 */
template <typename Dtype>
class Cifar10Dataset: public Dataset {
  // Public types
 public:
  typedef Dtype   Type;
  typedef Dataset Parent;


  // Public structures
 public:
  /*!
   *  \struct Image
   *  \brief  Cifar10 image = image bytes + label (10 classes)
   */
  struct Image {
    std::vector<Dtype> image;   // Image
    uint8_t            label;   // Label
  };


  // Protected attributes
 protected:
  bool               gray_;     // Grayscale the input?
  std::vector<Image> train_;    // Training set
  std::vector<Image> test_;     // Testing set


  // Protected methods
 protected:
  /*!
   * Get the cifar10 image width.
   *
   *  \return Cifar10 image width
   */
  static uint32_t Cifar10ImageWidth() {
    return 32;
  }

  /*!
   * Get the cifar10 image height.
   *
   *  \return Cifar10 image height
   */
  static uint32_t Cifar10ImageHeight() {
    return 32;
  }

  /*!
   * Get the cifar10 image channel.
   *
   *  \return Cifar10 image channel
   */
  static uint32_t Cifar10ImageChannel() {
    return 3;
  }

  /*!
   * Read a cifar10 dataset.
   * Check here for the dataset format:
   * http://www.cs.toronto.edu/~kriz/cifar.html
   *
   *  \param[in]  dataset_file: file containing the dataset
   *
   *  \param[out] dataset     : cifar10 dataset
   *  \return     Error?
   */
  bool ReadDataset(const char* dataset_file,
                   std::vector<Image>* dataset) {
    // Open the images file
    std::FILE* fp = std::fopen(dataset_file, "rb");
    if (!fp) {
      Report(kError, "Can't open file '%s'", dataset_file);
      return false;
    }

    std::fseek(fp, 0, SEEK_END);
    size_t file_size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    // Size of a cifar image
    uint32_t cifar10_image_size = Cifar10ImageWidth()  *
                                  Cifar10ImageHeight() *
                                  Cifar10ImageChannel();

    // Size of an image
    size_t image_size = ImageWidth() * ImageHeight() * ImageChannel();

    // Size of a label
    size_t label_size = 1;

    // Number of images
    size_t image_count =
      file_size / ((label_size + cifar10_image_size) * sizeof(uint8_t));

    // Number of channels (diff)
    size_t num_channel_diff = cifar10_image_size / image_size;

    // Add the images to the dataset
    size_t index = dataset->size();
    dataset->resize(index + image_count);

    // Read all the images
    std::vector<uint8_t> buffer(cifar10_image_size);
    for (size_t i = 0; i < image_count; ++i) {
      Image& img = (*dataset)[index + i];
      uint32_t size_to_read = label_size * sizeof(uint8_t);
      if (std::fread(&img.label, 1, size_to_read, fp) != size_to_read) {
        Report(kError, "Can't read images in '%s'", dataset_file);
        std::fclose(fp);
        return false;
      }
      // Check the label is between [0, 9]
      if (img.label > 9) {
        Report(kError, "Invalid label %d in file '%s'",
               img.label, dataset_file);
        std::fclose(fp);
        return false;
      }
      size_to_read = cifar10_image_size * sizeof(uint8_t);
      if (std::fread(&buffer[0], 1, size_to_read, fp) != size_to_read) {
        Report(kError, "Can't read images in '%s'", dataset_file);
        std::fclose(fp);
        return false;
      }
      img.image.resize(image_size);
      uint32_t buffer_index = 0;
      for (size_t j = 0; j < image_size; ++j) {
        if (num_channel_diff == 1) {
          img.image[j] = Dtype(buffer[buffer_index++]);
        } else if (num_channel_diff == 3) {
          // Convert RGB to grayscale (luminosity)
          img.image[j] = Dtype(0.2126 * buffer[buffer_index + 0]) +
                         Dtype(0.7152 * buffer[buffer_index + 1]) +
                         Dtype(0.0722 * buffer[buffer_index + 2]);
          buffer_index += 3;
        } else {
          Report(kError, "Unknown image depth in '%s'", dataset_file);
          std::fclose(fp);
          return false;
        }
        img.image[j] /= 0xFF;
      }
    }

    std::fclose(fp);
    return true;
  }


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  gray: grayscale the input?
   */
  explicit Cifar10Dataset(bool gray) {
    gray_ = gray;
  }

  /*!
   * Destructor.
   */
  virtual ~Cifar10Dataset() {}

  /*!
   * Get the image width.
   *
   *  \return Image width
   */
  static uint32_t ImageWidth() {
    return Cifar10ImageWidth();
  }

  /*!
   * Get the image height.
   *
   *  \return Image height
   */
  static uint32_t ImageHeight() {
    return Cifar10ImageHeight();
  }

  /*!
   * Get the image channel.
   *
   *  \return Image channel
   */
  uint32_t ImageChannel() const {
    if (gray_) {
      return 1;
    }
    return Cifar10ImageChannel();
  }

  /*!
   * Get the number of classes.
   *
   *  \return Number of classes
   */
  static uint32_t NumClass() {
    return 10;
  }

  /*!
   * Load a cifar10 dataset.
   *
   *  \param[in]  dataset_path: path to the dataset
   *  \param[in]  prefix      : dataset prefix
   *
   *  \param[out] dataset     : cifar10 dataset
   *  \return     Error?
   */
  bool LoadDataset(const char* dataset_path, const char* prefix,
                   std::vector<Image>* dataset) {
    bool res;
    std::string dataset_file = std::string(dataset_path) + "/" +
                               std::string(prefix) + "_batch.bin";
    struct stat buffer;
    if (stat(dataset_file.c_str(), &buffer)) {
      res          = true;
      size_t index = 1;
      while (true) {
        dataset_file = std::string(dataset_path) + "/" + std::string(prefix) +
                       "_batch_" + std::to_string(index) + ".bin";
        if (stat(dataset_file.c_str(), &buffer)) {
          break;
        }
        res = ReadDataset(dataset_file.c_str(), dataset) && res;
        ++index;
      }
    } else {
      res = ReadDataset(dataset_file.c_str(), dataset);
    }
    return res;
  }

  /*!
   * Load the dataset.
   *
   *  \param[in]  dataset_path: path to the dataset root directory
   *
   *  \return     Error?
   */
  virtual bool Load(const char* dataset_path) {
    // Clear datasets
    train_.clear();
    test_.clear();

    const char* path = std::strtok(const_cast<char*>(dataset_path), ":");
    while (path) {
      Report(kInfo, "Loading dataset from '%s'", path);
      std::string spath = std::string(path);

      // Training set
      if (!LoadDataset(spath.c_str(), "data", &train_) ||
          !LoadDataset(spath.c_str(), "test", &test_)) {
        return false;
      }

      // Go to next path
      path = std::strtok(nullptr, ":");
    }

    // Randomly shuffle the dataset to have uniform mini-batches with a good
    // estimation of the gradient: we want each mini-batch gradient to be very
    // close to the batch (dataset) gradient
    std::random_device rd;
    std::default_random_engine re(rd());
    std::shuffle(train_.begin(), train_.end(), re);
    std::shuffle(test_.begin() , test_.end() , re);

    return true;
  }

  /*!
   * Get the training set.
   *
   *  \return Training set
   */
  const std::vector<Image>& Train() const {
    return train_;
  }

  /*!
   * Get the testing set.
   *
   *  \return Testing set
   */
  const std::vector<Image>& Test() const {
    return test_;
  }
};


/*!
 *  \class  Cifar10DataLayer
 *  \brief  Cifar10 data layer
 */
template <typename Dtype>
class Cifar10DataLayer: public LayerData<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerData<Dtype>  Parent;


  // Protected attributes
 protected:
  Cifar10Dataset<Dtype> dataset_;               // Cifar10 dataset
  uint32_t              dataset_train_index_;   // Dataset index (training)
  uint32_t              dataset_test_index_;    // Dataset index (testing)


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  param: parameters
   *  \param[in]  gray : grayscale the input?
   */
  Cifar10DataLayer(const char* name, const Param& param, bool gray):
    LayerData<Dtype>(name), dataset_(gray) {
    // Parameters
    std::string dataset_path;
    uint32_t batch_size;
    param.Get("dataset_path", &dataset_path);
    param.Get("batch_size"  , &batch_size);

    if (!dataset_.Load(dataset_path.c_str())) {
      return;
    }

    Report(kInfo, "Training set: %ld image(s)", dataset_.Train().size());
    Report(kInfo, "Testing  set: %ld image(s)", dataset_.Test().size());

    // Set index at the beginning of the dataset
    dataset_train_index_ = dataset_test_index_ = 0;

    // Create 2 outputs: images and labels
    // There's no gradient for the labels as we don't backpropagate them
    Parent::out_.resize(2);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(dataset_.ImageWidth(),
                                           dataset_.ImageHeight(),
                                           dataset_.ImageChannel(),
                                           batch_size);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(1, 1, 1, batch_size, false);
  }

  /*!
   * Destructor.
   */
  virtual ~Cifar10DataLayer() {}

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
    uint32_t dataset_test_size = uint32_t(dataset_.Test().size());
    if (!dataset_test_size) {
      // No dataset: we are done
      return true;
    }

    // We are done if we are at the end of the testing dataset
    bool testing_done = dataset_test_index_ >= dataset_test_size;

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
    const std::vector<typename Cifar10Dataset<Dtype>::Image>* dataset;
    uint32_t* dataset_index;
    if (state.phase == State::PHASE_TRAIN) {
      dataset       = &dataset_.Train();
      dataset_index = &dataset_train_index_;
    } else {
      dataset       = &dataset_.Test();
      dataset_index = &dataset_test_index_;
    }

    if (dataset->empty()) {
      Report(kError, "Empty dataset");
      Parent::out_[0]->Zero();
      Parent::out_[1]->Zero();
      return;
    }

    if (*dataset_index >= uint32_t(dataset->size())) {
      Report(kError, "Invalid dataset index");
      Parent::out_[0]->Zero();
      Parent::out_[1]->Zero();
      return;
    }

    Dtype* image_data = Parent::out_[0]->Data();
    Dtype* label_data = Parent::out_[1]->Data();

    uint32_t image_size = Parent::out_[0]->size[0] * Parent::out_[0]->size[1] *
                          Parent::out_[0]->size[2];
    uint32_t batch_size = Parent::out_[0]->size[3];

    bool testing_done = false;

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
      // Current image
      const typename Cifar10Dataset<Dtype>::Image& image =
        (*dataset)[*dataset_index];

      // Copy the pixels
      std::memcpy(image_data + batch * image_size,
                  &image.image[0], image_size * sizeof(Dtype));

      // Copy the labels
      label_data[batch] = image.label;

      // Go to the next image
      if (++*dataset_index >= uint32_t(dataset->size())) {
        if (state.phase == State::PHASE_TRAIN) {
          // Rewind
          *dataset_index = 0;
        } else {
          // Clamp
          *dataset_index = uint32_t(dataset->size()) - 1;
          testing_done   = true;
        }
      }
    }

    if (testing_done) {
      // Mark the testing dataset as done
      *dataset_index = uint32_t(dataset->size());
    }
  }
};


/*!
 *  \class  Cifar10Model
 *  \brief  Cifar10 model
 */
template <typename Dtype>
class Cifar10Model: public Model<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Model<Dtype>  Parent;


  // Protected attributes
 protected:
  std::shared_ptr<Mat<Dtype>> label_;   // Labels
  std::shared_ptr<Mat<Dtype>>  prob_;   // Probabilities


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name        : model name
   *  \param[in]  dataset_path: path to the dataset
   *  \param[in]  num_output  : matrix size (number of classes)
   *  \param[in]  batch_size  : matrix size (batch size)
   *  \param[in]  gray : grayscale the input?
   *  \param[in]  use_bn      : use batch norm?
   */
  Cifar10Model(const char* name, const char* dataset_path, uint32_t num_output,
               uint32_t batch_size, bool gray, bool use_bn):

  Model<Dtype>(name) {
    // Network architecture:
    //
    // DATA1 (INPUT)
    //   |
    //   |
    // CONV1
    //   |
    //   |
    // POOL1
    //   |
    //   |
    // RELU1
    //   |
    //   |
    // (BN1)
    //   |
    //   |
    // (SCALE1)
    //   |
    //   |
    // CONV2
    //   |
    //   |
    // RELU2
    //   |
    //   |
    // POOL2
    //   |
    //   |
    // (BN2)
    //   |
    //   |
    // (SCALE2)
    //   |
    //   |
    // CONV3
    //   |
    //   |
    // RELU3
    //   |
    //   |
    // POOL3
    //   |
    //   |
    // IP1
    //   |
    //   |
    // LOSS (OUTPUT)

    // Input layer parameters
    Param data_param;
    data_param.Add("dataset_path", dataset_path);
    data_param.Add("batch_size"  , batch_size);

    // Input layer
    std::vector<std::shared_ptr<Mat<Dtype>>> out = Parent::Add(
      std::make_shared<Cifar10DataLayer<Dtype>>("data1", data_param, gray));

    // Model input (images) and labels
    Parent::in_ = out[0];
    label_      = out[1];

    // Output of previous layer
    Parent::out_ = Parent::in_;

    // Conv layer parameters
    Param conv_param;
    conv_param.Add("num_output"   , 32);
    conv_param.Add("filter_width" , 3);
    conv_param.Add("filter_height", 3);
    conv_param.Add("padding_x"    , 1);
    conv_param.Add("padding_y"    , 1);
    conv_param.Add("stride_x"     , 1);
    conv_param.Add("stride_y"     , 1);

    // Pool layer parameters
    Param pool_param;
    pool_param.Add("filter_width" , 3);
    pool_param.Add("filter_height", 3);
    pool_param.Add("padding_x"    , 1);
    pool_param.Add("padding_y"    , 1);
    pool_param.Add("stride_x"     , 2);
    pool_param.Add("stride_y"     , 2);

    // Conv1, Pool1, Relu1
    Parent::out_ = Parent::Add(std::make_shared<LayerConv<Dtype>>("conv1",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      conv_param))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerPoolMax<Dtype>>("pool1",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      pool_param))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu1",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];

    // BN layer parameters
    Param bn_param;
    bn_param.Add("moving_avg_frac", 0.99f);
    if (use_bn) {
      // BN1, Scale1 (whitening activations)
      Parent::out_ = Parent::Add(std::make_shared<LayerBatchNorm<Dtype>>("bn1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        bn_param))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerScale<Dtype>>("scale1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        Param()))[0];
    }

    // Conv2, Relu2, Pool2
    Parent::out_ = Parent::Add(std::make_shared<LayerConv<Dtype>>("conv2",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      conv_param))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu2",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerPoolMax<Dtype>>("pool2",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      pool_param))[0];

    if (use_bn) {
      // BN2, Scale2 (whitening activations)
      Parent::out_ = Parent::Add(std::make_shared<LayerBatchNorm<Dtype>>("bn2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        bn_param))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerScale<Dtype>>("scale2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        Param()))[0];
    }

    // Increase the depth for the next conv layer
    conv_param.Add("num_output", 64);

    // Conv3, Relu3, Pool3
    Parent::out_ = Parent::Add(std::make_shared<LayerConv<Dtype>>("conv3",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      conv_param))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu3",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];
    Parent::out_ = Parent::Add(std::make_shared<LayerPoolMax<Dtype>>("pool3",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      pool_param))[0];

    // IP layer parameters
    Param ip_param;
    ip_param.Add("num_output", num_output);

    // IP1
    Parent::out_ = Parent::Add(std::make_shared<LayerInnerProduct<Dtype>>(
      "ip1", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
      ip_param))[0];

    // Loss (softmax)
    const std::vector<std::shared_ptr<Mat<Dtype>>>& softmax_out =
    Parent::Add(std::make_shared<LayerSoftMaxLoss<Dtype>>("loss",
      std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_,
      label_}));
    Parent::out_ = softmax_out[0];
    prob_        = softmax_out[1];
  }

  /*!
   * Destructor.
   */
  virtual ~Cifar10Model() {}

  /*!
   * Graph testing (inference).
   *
   *  \return     Accuracy
   */
  virtual Dtype Test() {
    // Create the state
    State state(State::PHASE_TEST);

    // Number of classes (outputs)
    uint32_t num_output = prob_->size[2];

    // Get the data layer to keep track of the testing index
    std::shared_ptr<Cifar10DataLayer<Dtype>> cifar10_data =
      std::dynamic_pointer_cast<Cifar10DataLayer<Dtype>>(Parent::DataLayer());
    if (!cifar10_data) {
      Report(kError, "No data layer found in model '%s'", Parent::Name());
      return Dtype(0);
    }

    uint32_t step = 0;
    Dtype acc     = Dtype(0);
    while (!cifar10_data->TestingDone()) {
      // Current test index
      uint32_t index = cifar10_data->TestIndex();

      // Inference
      Parent::Forward(state);

      // Actual batch size = where we are - where we were
      uint32_t actual_batch_size = cifar10_data->TestIndex() - index;

      // Prediction
      uint32_t pred = 0;
      for (uint32_t batch = 0; batch < actual_batch_size; ++batch) {
        // Outputs
        const Dtype* data = prob_->Data() + batch * num_output;

        // Network prediction
        uint32_t predicted_number = 0;
        for (uint32_t number = 0; number < num_output; ++number) {
          if (data[predicted_number] < data[number]) {
            predicted_number = number;
          }
        }

        // Check if the prediction is correct
        if (uint32_t(label_->Data()[batch]) == predicted_number) {
          ++pred;
        }
      }

      // Go to next step and accumulate the accuracy
      ++step;
      acc += Dtype(pred) / actual_batch_size;
    }

    // Overall accuracy
    return acc / step;
  }
};


}  // namespace jik


int main(int argc, char* argv[]) {
  using namespace jik;  // NOLINT(build/namespaces)

  // 32-bit float quantization
  typedef float Dtype;

  // Hyperparameters
  ArgParse arg(argc, argv);
  const char* dataset_path = arg.Arg("-dataset");
  const char* model_path   = arg.Arg("-model");
  const char* model_name   = arg.Arg("-name");
  const char* solver_type  = arg.Arg("-solver");
  bool        train        = arg.ArgExists("-train");
  bool        gray         = arg.ArgExists("-gray");
  bool        use_bn       = arg.ArgExists("-bn");
  uint32_t batch_size;
  Dtype learning_rate, decay_rate, momentum, reg, clip, lr_scale;
  uint32_t num_step, print_each, test_each, save_each, lr_scale_each;
  arg.Arg<uint32_t>("-batchsize"  , 128          , &batch_size);
  arg.Arg<Dtype>   ("-lr"         , Dtype(0.0005), &learning_rate);
  arg.Arg<Dtype>   ("-decayrate"  , Dtype(0.999) , &decay_rate);
  arg.Arg<Dtype>   ("-momentum"   , Dtype(0.9)   , &momentum);
  arg.Arg<Dtype>   ("-reg"        , Dtype(0.001) , &reg);
  arg.Arg<Dtype>   ("-clip"       , Dtype(5)     , &clip);
  arg.Arg<uint32_t>("-numstep"    , 50000        , &num_step);
  arg.Arg<uint32_t>("-printeach"  , 100          , &print_each);
  arg.Arg<uint32_t>("-testeach"   , 1000         , &test_each);
  arg.Arg<uint32_t>("-saveeach"   , 1000         , &save_each);
  arg.Arg<uint32_t>("-lrscaleeach", 10000        , &lr_scale_each);
  arg.Arg<Dtype>   ("-lrscale"    , Dtype(0.1)   , &lr_scale);

  if (!dataset_path || (!train && !model_path) || arg.ArgExists("-h")) {
    Report(kInfo, "Usage: %s -dataset <path/to/cifar10/dataset> [-train] "
           "[-model <path/to/cifar10/model>] [-gray] [-bn]", argv[0]);
    return -1;
  }

  // Default model and solver names
  if (!model_name) {
    model_name = "cifar10";
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

  // Create the model
  Cifar10Model<Dtype> model(model_name, dataset_path,
                            Cifar10Dataset<Dtype>::NumClass(),
                            batch_size, gray, use_bn);

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

  return 0;
}
