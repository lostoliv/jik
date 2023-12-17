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
 *  \class  MnistDataset
 *  \brief  Mnist dataset
 */
template <typename Dtype>
class MnistDataset: public Dataset {
  // Public types
 public:
  typedef Dtype   Type;
  typedef Dataset Parent;


  // Public structures
 public:
  /*!
   *  \struct Image
   *  \brief  Mnist image = image bytes + label (10 classes: 0 to 9 number)
   */
  struct Image {
    std::vector<Dtype> image;         // Image
    uint8_t            label;         // Label
  };


  // Protected attributes
 protected:
  uint32_t           image_width_;    // Image width
  uint32_t           image_height_;   // Image height
  std::vector<Image> train_;          // Training set
  std::vector<Image> test_;           // Testing set


  // Protected methods
 protected:
  /*!
   * Switch the endianness of a 32-bits number.
   *
   *  \param[in]  num: 32-bits number
   *
   *  \return     32-bits number with endianness switched
   */
  uint32_t SwapEndian32(uint32_t num) {
    return (num & 0x000000FF) << 24 |
           (num & 0x0000FF00) << 8  |
           (num & 0x00FF0000) >> 8  |
           (num & 0xFF000000) >> 24;
  }

  /*!
   * Read a mnist dataset.
   * Check here for the dataset format: http://yann.lecun.com/exdb/mnist/
   *
   *  \param[in]  file_image: file containing the images
   *  \param[in]  file_label: file containing the labels
   *
   *  \param[out] dataset   : mnist dataset
   *  \return     Error?
   */
  bool ReadDataset(const std::string& file_image,
                   const std::string& file_label,
                   std::vector<Image>* dataset) {
    // File header constants
    const uint32_t kMnistImageHeader = 0x803;
    const uint32_t kMnistLabelHeader = 0x801;

    // Open the images file
    std::FILE* fp = std::fopen(file_image.c_str(), "rb");
    if (!fp) {
      Report(kError, "Can't open file '%s'", file_image.c_str());
      return false;
    }

    // Read the images file header
    uint32_t magic, image_count, image_rows, image_columns;
    size_t size_to_read = sizeof(uint32_t);
    if (std::fread(&magic        , 1, size_to_read, fp) != size_to_read ||
        std::fread(&image_count  , 1, size_to_read, fp) != size_to_read ||
        std::fread(&image_rows   , 1, size_to_read, fp) != size_to_read ||
        std::fread(&image_columns, 1, size_to_read, fp) != size_to_read) {
      Report(kError, "Invalid file header in '%s'", file_image.c_str());
      std::fclose(fp);
      return false;
    }
    magic         = SwapEndian32(magic);
    image_count   = SwapEndian32(image_count);
    image_rows    = SwapEndian32(image_rows);
    image_columns = SwapEndian32(image_columns);

    // Check the magic number
    if (magic != kMnistImageHeader) {
      Report(kError, "Invalid file format in '%s'", file_image.c_str());
      std::fclose(fp);
      return false;
    }

    // Set or check the image width
    if (image_width_) {
      if (image_rows != image_width_) {
        Report(kError, "Invalid image format in '%s'", file_image.c_str());
        std::fclose(fp);
        return false;
      }
    } else {
      image_width_ = image_rows;
    }

    // Set or check the image height
    if (image_height_) {
      if (image_columns != image_height_) {
        Report(kError, "Invalid image format in '%s'", file_image.c_str());
        std::fclose(fp);
        return false;
      }
    } else {
      image_height_ = image_columns;
    }

    // Size of an image
    uint32_t mnist_size = image_rows * image_columns;

    // Read all the images
    std::vector<uint8_t> images(image_count * mnist_size);
    size_to_read = image_count * mnist_size * sizeof(uint8_t);
    if (std::fread(&images[0], 1, size_to_read, fp) != size_to_read) {
      Report(kError, "Can't read images in '%s'", file_image.c_str());
      std::fclose(fp);
      return false;
    }
    std::fclose(fp);

    // Open the labels file
    fp = std::fopen(file_label.c_str(), "rb");
    if (!fp) {
      Report(kError, "Can't open file '%s'", file_label.c_str());
      return false;
    }

    // Read the label file header
    uint32_t label_count;
    size_to_read = sizeof(uint32_t);
    if (std::fread(&magic      , 1, size_to_read, fp) != size_to_read ||
        std::fread(&label_count, 1, size_to_read, fp) != size_to_read) {
      Report(kError, "Invalid file header in '%s'", file_label.c_str());
      std::fclose(fp);
      return false;
    }
    magic       = SwapEndian32(magic);
    label_count = SwapEndian32(label_count);

    // Check the magic number and the number of labels
    // (must match the number of images)
    if (magic != kMnistLabelHeader || label_count != image_count) {
      Report(kError, "Invalid file format in '%s'", file_label.c_str());
      std::fclose(fp);
      return false;
    }

    // Read all the labels
    std::vector<uint8_t> labels(image_count);
    size_to_read = image_count * sizeof(uint8_t);
    if (std::fread(&labels[0], 1, size_to_read, fp) != size_to_read) {
      Report(kError, "Can't read labels in '%s'", file_label.c_str());
      std::fclose(fp);
      return false;
    }
    std::fclose(fp);

    // Add the images to the dataset
    size_t index = dataset->size();
    dataset->resize(index + image_count);

    // Create the dataset
    for (uint32_t i = 0; i < image_count; ++i) {
      Image& img = (*dataset)[index + i];

      // Save the image
      img.image.resize(mnist_size);
      uint32_t image_index = i * mnist_size;
      for (uint32_t j = 0; j < image_columns; ++j) {
        for (uint32_t k = 0; k < image_rows; ++k) {
          uint32_t index = j * image_columns + k;
          img.image[index] = Dtype(images[image_index + index]) / 0xFF;
        }
      }

      // Save the label
      img.label = labels[i];
      // Check the label is between [0, 9]
      if (img.label > 9) {
        Report(kError, "Invalid label %d in file '%s'",
               img.label, file_label.c_str());
        return false;
      }
    }

    return true;
  }


  // Public methods
 public:
  /*!
   * Default constructor.
   */
  MnistDataset() {
    image_width_ = image_height_ = 0;
  }

  /*!
   * Destructor.
   */
  virtual ~MnistDataset() {}

  /*!
   * Get the image width.
   *
   *  \return Image width
   */
  uint32_t ImageWidth() const {
    return image_width_;
  }

  /*!
   * Get the image height.
   *
   *  \return Image height
   */
  uint32_t ImageHeight() const {
    return image_height_;
  }

  /*!
   * Get the image channel.
   *
   *  \return Image channel
   */
  static uint32_t ImageChannel() {
    return 1;
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
      std::string train_file_image(spath + "/train-images-idx3-ubyte");
      std::string train_file_label(spath + "/train-labels-idx1-ubyte");
      if (!ReadDataset(train_file_image, train_file_label, &train_)) {
        return false;
      }

      // Testing set
      std::string test_file_image(spath + "/t10k-images-idx3-ubyte");
      std::string test_file_label(spath + "/t10k-labels-idx1-ubyte");
      if (!ReadDataset(test_file_image, test_file_label, &test_)) {
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
 *  \class  MnistDataLayer
 *  \brief  Mnist data layer
 */
template <typename Dtype>
class MnistDataLayer: public LayerData<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerData<Dtype>  Parent;


  // Protected attributes
 protected:
  MnistDataset<Dtype> dataset_;               // Mnist dataset
  uint32_t            dataset_train_index_;   // Dataset index (training)
  uint32_t            dataset_test_index_;    // Dataset index (testing)


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  param: parameters
   */
  MnistDataLayer(const char* name, const Param& param):
    LayerData<Dtype>(name) {
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
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(
      dataset_.ImageWidth(), dataset_.ImageHeight(),
      dataset_.ImageChannel(), batch_size);
    Parent::out_[1] = std::make_shared<Mat<Dtype>>(1, 1, 1, batch_size, false);
  }

  /*!
   * Destructor.
   */
  virtual ~MnistDataLayer() {}

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
    const std::vector<typename MnistDataset<Dtype>::Image>* dataset;
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
      const typename MnistDataset<Dtype>::Image& image =
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
 *  \class  MnistModel
 *  \brief  Mnist model
 */
template <typename Dtype>
class MnistModel: public Model<Dtype> {
  // Public types
 public:
  typedef Dtype         Type;
  typedef Model<Dtype>  Parent;


  // Protected attributes
 protected:
  std::shared_ptr<Mat<Dtype>> label_;   // Labels
  std::shared_ptr<Mat<Dtype>> prob_;    // Probabilities


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name        : model name
   *  \param[in]  dataset_path: path to the dataset
   *  \param[in]  num_output  : matrix size (number of classes)
   *  \param[in]  batch_size  : matrix size (batch size)
   *  \param[in]  use_fc      : use fully-connected network?
   *  \param[in]  use_bn      : use batch norm?
   */
  MnistModel(const char* name, const char* dataset_path, uint32_t num_output,
             uint32_t batch_size, bool use_fc, bool use_bn):
    Model<Dtype>(name) {
    // Input layer parameters
    Param data_param;
    data_param.Add("dataset_path", dataset_path);
    data_param.Add("batch_size"  , batch_size);

    // Input layer
    std::vector<std::shared_ptr<Mat<Dtype>>> out = Parent::Add(
      std::make_shared<MnistDataLayer<Dtype>>("data1", data_param));

    // Model input (images) and labels
    Parent::in_ = out[0];
    label_      = out[1];

    // Output of previous layer
    Parent::out_ = Parent::in_;

    if (use_fc) {
      // Network architecture:
      //
      // DATA1 (INPUT)
      //   |
      //   |
      // IP1
      //   |
      //   |
      // RELU1
      //   |
      //   |
      // IP2
      //   |
      //   |
      // RELU2
      //   |
      //   |
      // IP3
      //   |
      //   |
      // LOSS (OUTPUT)

      // Hidden IP layer parameters
      // We use 64 hidden units
      Param ip_hidden_param;
      ip_hidden_param.Add("num_output", 64);

      // IP1
      Parent::out_ = Parent::Add(std::make_shared<LayerInnerProduct<Dtype>>(
        "ip1", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
        Parent::out_}, ip_hidden_param))[0];

      // Relu1
      Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];

      // IP2
      Parent::out_ = Parent::Add(std::make_shared<LayerInnerProduct<Dtype>>(
        "ip2", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
        Parent::out_}, ip_hidden_param))[0];

      // Relu2
      Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];

      // IP3
      // Final IP layer parameters
      Param ip_final_param;
      ip_final_param.Add("num_output", num_output);
      Parent::out_ = Parent::Add(std::make_shared<LayerInnerProduct<Dtype>>(
        "ip3", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
        Parent::out_}, ip_final_param))[0];
    } else {
      // Network architecture:
      //
      // DATA1 (INPUT)
      //   |
      //   |
      // CONV1
      //   |
      //   |
      // RELU1
      //   |
      //   |
      // POOL1
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
      // IP1
      //   |
      //   |
      // LOSS (OUTPUT)

      // Conv layer parameters
      Param conv_param;
      conv_param.Add("num_output"   , 8);
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

      // Conv1, Relu1, Pool1
      Parent::out_ = Parent::Add(std::make_shared<LayerConv<Dtype>>("conv1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        conv_param))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerPoolMax<Dtype>>("pool1",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        pool_param))[0];

      if (use_bn) {
        // BN layer parameters
        Param bn_param;
        bn_param.Add("moving_avg_frac", 0.99f);

        // BN1, Scale1 (whitening activations)
        Parent::out_ = Parent::Add(std::make_shared<LayerBatchNorm<Dtype>>(
          "bn1", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
          Parent::out_}, bn_param))[0];
        Parent::out_ = Parent::Add(std::make_shared<LayerScale<Dtype>>(
          "scale1", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
          Parent::out_}, Param()))[0];
      }

      // Increase the depth for the next conv layer
      conv_param.Add("num_output", 16);

      // Conv2, Relu2, Pool2
      Parent::out_ = Parent::Add(std::make_shared<LayerConv<Dtype>>("conv2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        conv_param))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerRelu<Dtype>>("relu2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_}))[0];
      Parent::out_ = Parent::Add(std::make_shared<LayerPoolMax<Dtype>>("pool2",
        std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        pool_param))[0];

      // IP layer parameters
      Param ip_param;
      ip_param.Add("num_output", num_output);

      // IP1
      Parent::out_ = Parent::Add(std::make_shared<LayerInnerProduct<Dtype>>(
        "ip1", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
        Parent::out_}, ip_param))[0];
    }

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
  virtual ~MnistModel() {}

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
    std::shared_ptr<MnistDataLayer<Dtype>> mnist_data =
      std::dynamic_pointer_cast<MnistDataLayer<Dtype>>(Parent::DataLayer());
    if (!mnist_data) {
      Report(kError, "No data layer found in model '%s'", Parent::Name());
      return Dtype(0);
    }

    uint32_t step = 0;
    Dtype acc     = Dtype(0);
    while (!mnist_data->TestingDone()) {
      // Current test index
      uint32_t index = mnist_data->TestIndex();

      // Inference
      Parent::Forward(state);

      // Actual batch size = where we are - where we were
      uint32_t actual_batch_size = mnist_data->TestIndex() - index;

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
  bool        use_fc       = arg.ArgExists("-fc");
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
    Report(kInfo, "Usage: %s -dataset <path/to/mnist/dataset> [-train] "
           "[-model <path/to/mnist/model>] [-fc] [-bn]", argv[0]);
    return -1;
  }

  // Default model and solver names
  if (!model_name) {
    model_name = "mnist";
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
  MnistModel<Dtype> model(model_name, dataset_path,
                          MnistDataset<Dtype>::NumClass(),
                          batch_size, use_fc, use_bn);

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
