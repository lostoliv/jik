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
#include <core/layer_eltwise_scale.h>
#include <core/layer_softmax_loss.h>
#include <core/solver_sgd.h>
#include <core/solver_rmsprop.h>
#include <recurrent/rnn.h>
#include <recurrent/lstm.h>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <random>


namespace jik {


/*!
 *  \class  TextgenDataset
 *  \brief  Textgen dataset
 */
class TextgenDataset: public Dataset {
  // Public types
 public:
  typedef Dataset Parent;


  // Protected attributes
 protected:
  std::vector<std::string> sentence_;         // List of sentences
  std::set<char>           vocab_;            // Vocabulary
  std::map<char, uint32_t> letter_to_index_;  // Mapping letters to indices
  std::map<uint32_t, char> index_to_letter_;  // Mapping indices to letters


  // Protected methods
 protected:
   /*!
    * Cleaning a string.
    * Removing leading and trailing spaces and tabs.
    *
    * \param[in]  str: string
    */
  static void CleanString(std::string* str) {
    // Trim leading spaces and tabs
    size_t startpos = str->find_first_not_of(" \t");
    if (std::string::npos != startpos) {
      *str = str->substr(startpos);
    }

    // Trim trailing spaces and tabs
    size_t endpos = str->find_last_not_of(" \t");
    if (std::string::npos != endpos) {
      *str = str->substr(0, endpos + 1);
    }
  }


  // Public methods
 public:
  /*!
   * Default constructor.
   */
  TextgenDataset() {}

  /*!
   * Destructor.
   */
  virtual ~TextgenDataset() {}

  /*!
   * Load the dataset.
   *
   *  \param[in]  dataset_path: path to the dataset root directory
   *
   *  \return     Error?
   */
  virtual bool Load(const char* dataset_path) {
    Report(kInfo, "Loading dataset '%s'", dataset_path);

    std::string line;
    std::ifstream fp(dataset_path);
    if (fp.fail()) {
      Report(kError, "Can't open file '%s'", dataset_path);
      return false;
    }

    while (std::getline(fp, line)) {
      CleanString(&line);
      if (line.empty()) {
        continue;
      }
      for (uint32_t i = 0; i < line.length(); ++i) {
        vocab_.insert(line[i]);
      }
      sentence_.push_back(line);
    }

    // Reserve index 0
    uint32_t i = 1;
    for (auto it = vocab_.begin(); it != vocab_.end(); ++it, ++i) {
      letter_to_index_[*it] = i;
      index_to_letter_[i]   = *it;
    }

    return true;
  }

  /*!
   * Get the number of sentences.
   *
   *  \return Number of sentences
   */
  uint32_t SentenceSize() const {
    return uint32_t(sentence_.size());
  }

  /*!
   * Get the number of letters.
   *
   *  \return Number of letters
   */
  uint32_t VocabSize() const {
    return uint32_t(vocab_.size());
  }

  /*!
   * Get a given sentence.
   *
   *  \param[in]  index: sentence index
   *
   *  \return     Sentence
   */
  const std::string& Sentence(uint32_t index) const {
    return sentence_[index];
  }

  /*!
   * Convert a character to an index.
   *
   *  \param[in]  ch: character
   *
   *  \return     Index
   */
  uint32_t LetterToIndex(char ch) const {
    auto it = letter_to_index_.find(ch);
    if (it == letter_to_index_.end()) {
      return 0;
    }
    return it->second;
  }

  /*!
   * Convert an index to a letter.
   *
   *  \param[in]  index: index
   *
   *  \return     Letter
   */
  char IndexToLetter(uint32_t index) const {
    auto it = index_to_letter_.find(index);
    if (it == index_to_letter_.end()) {
      return '\0';
    }
    return it->second;
  }
};


/*!
 *  \class  TextgenDataLayer
 *  \brief  Textgen data layer
 */
template <typename Dtype>
class TextgenDataLayer: public LayerData<Dtype> {
  // Public types
 public:
  typedef Dtype             Type;
  typedef LayerData<Dtype>  Parent;


  // Protected attributes
 protected:
  TextgenDataset dataset_;              // Textgen dataset
  uint32_t       dataset_train_index_;  // Index in the dataset (training)
  uint32_t       dataset_test_index_;   // Index in the dataset (testing)
  uint32_t       num_predict_;          // Number of predictions
  std::string    sentence_;             // Currently loaded sentence


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name : layer name
   *  \param[in]  param: parameters
   */
  TextgenDataLayer(const char* name, const Param& param):
    Parent(name) {
    // Parameters
    std::string dataset_path;
    uint32_t batch_size;
    param.Get("dataset_path", &dataset_path);
    param.Get("num_predict" , &num_predict_);
    param.Get("batch_size"  , &batch_size);

    if (!dataset_.Load(dataset_path.c_str())) {
      return;
    }

    Report(kInfo, "Sentence   size: %ld", dataset_.SentenceSize());
    Report(kInfo, "Vocabulary size: %ld", dataset_.VocabSize());

    // Set index at the beginning of the dataset
    dataset_train_index_ = dataset_test_index_ = 0;

    // Create 1 output for the labels
    // There's no gradient as we don't backpropagate them
    Parent::out_.resize(1);
    Parent::out_[0] = std::make_shared<Mat<Dtype>>(1, 1, 1, batch_size, false);
  }

  /*!
   * Destructor.
   */
  virtual ~TextgenDataLayer() {}

  /*!
   * Get the dataset.
   *
   *  \return Dataset
   */
  const TextgenDataset& Dataset() const {
    return dataset_;
  }

  /*!
   * Get the prediction index (aka testing index).
   *
   *  \return Prediction index
   */
  uint32_t PredictionIndex() const {
    return dataset_test_index_;
  }

  /*!
   * Check if testing is done
   * (i.e. if we are at the end of the testing dataset).
   *
   *  \return Testing done?
   */
  bool TestingDone() {
    if (!num_predict_) {
      // No prediction: we are done
      return true;
    }

    // We are done if we did all the predictions
    bool testing_done = dataset_test_index_ == num_predict_;

    if (testing_done) {
      // If we are done, we rewind
      dataset_test_index_ = 0;
    }

    return testing_done;
  }

  /*!
   * Get the currently loaded sentence.
   *
   *  \return Currently loaded sentence
   */
  const std::string& Sentence() const {
    return sentence_;
  }

  /*!
   * Forward pass.
   *
   *  \param[in]  state: state
   */
  virtual void Forward(const State& state) {
    if (state.phase == State::PHASE_TEST) {
      if (dataset_test_index_ >= num_predict_) {
        Report(kError, "Invalid dataset index");
        return;
      }
      // During testing, just increase a counter
      ++dataset_test_index_;
      return;
    }

    uint32_t sentence_size = dataset_.SentenceSize();
    if (!sentence_size) {
      Report(kError, "Empty dataset");
      sentence_.clear();
      return;
    }
    if (dataset_train_index_ >= sentence_size) {
      Report(kError, "Invalid dataset index");
      sentence_.clear();
      return;
    }

    // Load the current sentence
    sentence_ = dataset_.Sentence(dataset_train_index_);

    // Go to the next sentence
    if (++dataset_train_index_ >= sentence_size) {
      dataset_train_index_ = 0;
    }
  }
};


/*!
 *  \class  TextgenModel
 *  \brief  Textgen recurrent model
 */
template <class R>
class TextgenModel: public R {
  // Protected types
 protected:
  // Public types
 public:
  typedef typename R::Type  Dtype;
  typedef Dtype             Type;
  typedef R                 Parent;


  // Protected attributes
 protected:
  std::shared_ptr<TextgenDataLayer<Dtype>> data_layer_;   // Data layer
  Dtype                                    temperature_;  // Temperature
  std::shared_ptr<Mat<Dtype>>              prob_;         // Probabilities


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name       : model name
   *  \param[in]  data_layer : data layer
   *  \param[in]  size_in    : input size
   *  \param[in]  hidden_size: hidden state size
   *  \param[in]  range      : value range ([-range/2, range/2])
   *  \param[in]  batch_size : batch size
   */
  TextgenModel(const char* name,
               const std::shared_ptr<TextgenDataLayer<Dtype>>& data_layer,
               Dtype temperature, uint32_t size_in,
               const std::vector<uint32_t>& hidden_size,
               Dtype range, uint32_t batch_size):
    R(name, size_in, hidden_size, data_layer->Dataset().VocabSize() + 1,
      range, batch_size) {
    data_layer_  = data_layer;
    temperature_ = temperature;
  }

  /*!
   * Destructor.
   */
  virtual ~TextgenModel() {}

  /*!
   * Create a data layer.
   *
   *  \param[in]  dataset_path: path to the dataset
   *  \param[in]  num_predict : number of predictions
   *  \param[in]  batch_size  : batch size
   *
   *  \return     Data layer
   */
  static std::shared_ptr<TextgenDataLayer<Dtype>> CreateDataLayer(
    const char* dataset_path, uint32_t num_predict, uint32_t batch_size) {
    Param param;
    param.Add("dataset_path", dataset_path);
    param.Add("num_predict" , num_predict);
    param.Add("batch_size"  , batch_size);
    return std::make_shared<TextgenDataLayer<Dtype>>("data1", param);
  }

  /*!
   * Create at a specific index.
   *
   *  \param[in]  index: data index
   */
  virtual void Create(uint32_t index) {
    Parent::Create(index);

    // Add a scale (temperature) layer
    if (temperature_ > std::numeric_limits<Dtype>::epsilon() &&
        temperature_ < Dtype(1) -
        std::numeric_limits<Dtype>::epsilon()) {
      Param param;
      param.Add("scale", temperature_);
      Parent::out_ = Parent::Add(std::make_shared<EltwiseScaleLayer<Dtype>>(
        "", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{Parent::out_},
        param))[0];
    }

    // Add a softmax layer
    std::shared_ptr<Mat<Dtype>> label = data_layer_->Output()[0];
    const std::vector<std::shared_ptr<Mat<Dtype>>>& out =
    Parent::Add(std::make_shared<LayerSoftMaxLoss<Dtype>>(
      "", std::initializer_list<std::shared_ptr<Mat<Dtype>>>{
      Parent::out_, label}));
    Parent::out_ = out[0];
    prob_        = out[1];
  }

  /*!
   * Graph training (forward + backward pass).
   *
   *  \return Loss
   */
  virtual Dtype Train() {
    // Clear the previous iteration state
    Parent::ClearPrevState();

    State state(State::PHASE_TRAIN);

    // Load the data
    data_layer_->Forward(state);

    // Get the label, sentence dataset and currently loaded sentence
    std::shared_ptr<Mat<Dtype>> label = data_layer_->Output()[0];
    const TextgenDataset& dataset     = data_layer_->Dataset();
    const std::string& sentence       = data_layer_->Sentence();

    uint32_t len = sentence.length();
    if (!len) {
      return Dtype(0);
    }

    Dtype loss = Dtype(0);
    for (uint32_t i = 0; i <= len; ++i) {
      uint32_t index_src = 0;
      uint32_t index_dst = 0;
      if (i) {
        index_src = dataset.LetterToIndex(sentence[i - 1]);
      }
      if (i != len) {
        index_dst = dataset.LetterToIndex(sentence[i]);
      }

      *label->Data() = index_dst;
      Create(index_src);
      loss += Parent::Train();
    }

    return loss / len;
  }

  /*!
   * Graph testing (inference).
   *
   *  \return Accuracy
   */
  virtual Dtype Test() {
    // Clear the previous iteration state
    Parent::ClearPrevState();

    State state(State::PHASE_TEST);

    // Get the sentence dataset
    const TextgenDataset& dataset = data_layer_->Dataset();

    // Max length of a predicted sentence
    const uint32_t kMaxSentenceLen = 80;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Dtype> dist(Dtype(0), Dtype(1));

    while (!data_layer_->TestingDone()) {
      // Load the data
      data_layer_->Forward(state);

      std::string sentence;
      while (true) {
        uint32_t index;
        if (sentence.empty())  {
          index = 0;
        } else {
          index = dataset.LetterToIndex(sentence[sentence.length() - 1]);
        }

        // Inference
        Create(index);
        Parent::Forward(state);

        // Pseudo-randomly choose an index
        index      = 0;
        Dtype r    = dist(gen);
        Dtype x    = Dtype(0);
        Dtype* out = prob_->Data();
        for (uint32_t i = 0; i < prob_->Size(); ++i) {
          x += out[i];
          if (x > r) {
            break;
          }
          ++index;
        }

        // End of the sentence predicted
        if (!index) {
          break;
        }

        // Add the character to the sentence
        sentence += dataset.IndexToLetter(index);

        // Too many characters, we stop
        if (sentence.length() > kMaxSentenceLen) {
          break;
        }
      }

      Report(kInfo, "Predicted sentence %ld: '%s'",
             data_layer_->PredictionIndex(), sentence.c_str());
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
  const char* dataset_path = arg.Arg("-dataset");
  const char* model_type   = arg.Arg("-model");
  const char* model_name   = arg.Arg("-name");
  const char* solver_type  = arg.Arg("-solver");
  Dtype learning_rate, decay_rate, momentum, reg,
        clip, lr_scale, temperature, range;
  uint32_t batch_size, num_step, print_each, test_each, save_each,
           lr_scale_each, num_predict, embed_size, hs;
  arg.Arg<uint32_t>("-batchsize"  , 128         , &batch_size);
  arg.Arg<Dtype>   ("-lr"         , Dtype(0.001), &learning_rate);
  arg.Arg<Dtype>   ("-decayrate"  , Dtype(0.999), &decay_rate);
  arg.Arg<Dtype>   ("-momentum"   , Dtype(0.9)  , &momentum);
  arg.Arg<Dtype>   ("-reg"        , Dtype(0.001), &reg);
  arg.Arg<Dtype>   ("-clip"       , Dtype(5)    , &clip);
  arg.Arg<uint32_t>("-numstep"    , 50000       , &num_step);
  arg.Arg<uint32_t>("-printeach"  , 100         , &print_each);
  arg.Arg<uint32_t>("-testeach"   , 1000        , &test_each);
  arg.Arg<uint32_t>("-saveeach"   , 1000        , &save_each);
  arg.Arg<uint32_t>("-lrscaleeach", 10000       , &lr_scale_each);
  arg.Arg<Dtype>   ("-lrscale"    , Dtype(0.1)  , &lr_scale);
  arg.Arg<Dtype>   ("-temperature", Dtype(1)    , &temperature);
  arg.Arg<uint32_t>("-numpredict" , 10          , &num_predict);
  arg.Arg<uint32_t>("-embedsize"  , 5           , &embed_size);
  arg.Arg<uint32_t>("-hs"         , 20          , &hs);
  arg.Arg<Dtype>   ("-range"      , Dtype(0.2)  , &range);

  if (!dataset_path || arg.ArgExists("-h")) {
    Report(kInfo, "Usage: %s -dataset <path/to/text/file> "
           "[-model <rnn/lstm>]", argv[0]);
    return -1;
  }

  // Default model type and model and solver names
  if (!model_type) {
    model_type = "rnn";
  }
  if (!model_name) {
    model_name = "textgen";
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
  Report(kInfo, "Temperature             : %f", temperature);
  Report(kInfo, "Number of predictions   : %d", num_predict);
  Report(kInfo, "Embedded size           : %d", embed_size);
  Report(kInfo, "Hidden size             : %d", hs);
  Report(kInfo, "Value range             : %f", range);

  // Create either a RNN or LSTM based recurrent model
  Model<Dtype>* model;
  if (!std::strcmp(model_type, "rnn")) {
    Report(kInfo, "Creating RNN model '%s'", model_name);
    model = new TextgenModel<Rnn<Dtype>>(model_name,
      TextgenModel<Rnn<Dtype>>::CreateDataLayer(dataset_path, num_predict,
      batch_size), temperature, embed_size, {hs, hs}, range, batch_size);
  } else if (!std::strcmp(model_type, "lstm")) {
    Report(kInfo, "Creating LSTM model '%s'", model_name);
    model = new TextgenModel<Lstm<Dtype>>(model_name,
      TextgenModel<Lstm<Dtype>>::CreateDataLayer(dataset_path, num_predict,
      batch_size), temperature, embed_size, {hs, hs}, range, batch_size);
  } else {
    Report(kError, "Unknown model type '%s'", model_type);
    return -1;
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
  if (!solver->Train(model, num_step, learning_rate)) {
    return -1;
  }

  // Clean
  delete model;
  delete solver;

  return 0;
}
