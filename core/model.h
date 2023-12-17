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


#ifndef CORE_MODEL_H_
#define CORE_MODEL_H_


#include <core/log.h>
#include <core/layer.h>
#include <core/layer_data.h>
#include <core/layer_loss.h>
#include <cstdio>
#include <memory>
#include <vector>
#include <string>


namespace jik {


/*!
 *  \class  Model
 *  \brief  Base model class
 *
 * A model is an execution graph composed of a
 * stack of layers with a defined input and output.
 */
template <typename Dtype>
class Model {
  // Public types
 public:
  typedef Dtype Type;


  // Protected attributes
 protected:
  std::string                                name_;   // Model name
  std::vector<std::shared_ptr<Layer<Dtype>>> layer_;  // List of layers
  std::shared_ptr<Mat<Dtype>>                in_;     // Input  of the model
  std::shared_ptr<Mat<Dtype>>                out_;    // Output of the model


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  name: graph name
   */
  explicit Model(const char* name) {
    Check(name && *name, "A graph must have a name");
    name_ = name;
  }

  /*!
   * Destructor.
   */
  virtual ~Model() {}

  /*!
   * Get the graph name.
   *
   *  \return Model name
   */
  const char* Name() const {
    return name_.c_str();
  }

  /*!
   * Clear the layers.
   */
  void Clear() {
    layer_.clear();
  }

  /*!
   * Get the batch size.
   *
   *  \return Batch size
   */
  uint32_t BatchSize() const {
    return in_->size[3];
  }

  /*!
   * Get the input of the model.
   *
   *  \return Input of the model
   */
  const std::shared_ptr<Mat<Dtype>>& In() const {
    return in_;
  }

  /*!
   * Get the output of the model.
   *
   *  \return Output of the model
   */
  const std::shared_ptr<Mat<Dtype>>& Out() const {
    return out_;
  }

  /*!
   * Get all the layers weights.
   *
   *  \param[out] weight: list of weights
   */
  virtual void GetWeight(std::vector<std::shared_ptr<Mat<Dtype>>>* weight) {
    weight->clear();
    for (size_t i = 0; i < layer_.size(); ++i) {
      layer_[i]->GetWeight(weight);
    }
  }

  /*!
   * Read the graph from a file stream.
   *
   *  \param[in]  fp: file stream
   *
   *  \return     Data size read from the file
   */
  size_t Read(std::FILE* fp) const {
    size_t res = 0;
    for (size_t i = 0; i < layer_.size(); ++i) {
      std::vector<std::shared_ptr<Mat<Dtype>>> weight;
      layer_[i]->GetWeight(&weight);
      for (const std::shared_ptr<Mat<Dtype>>& w : weight) {
        // Read the number of weights
        uint32_t weight_size;
        res += std::fread(reinterpret_cast<void*>(&weight_size), 1,
                          sizeof(uint32_t), fp);
        // Check the number of weights in the file is matching
        if (w->Size() != weight_size) {
          Report(kError, "Weights from file is not matching current model");
          return res;
        }
        // Read the weights
        res += std::fread(reinterpret_cast<void*>(w->Data()), 1,
                          sizeof(Dtype) * weight_size, fp);
      }
    }
    return res;
  }

  /*!
   * Write the graph in a file stream.
   *
   *  \param[in]  fp: file stream
   *
   *  \return     Data size written to the file
   */
  size_t Write(std::FILE* fp) const {
    size_t res = 0;
    for (size_t i = 0; i < layer_.size(); ++i) {
      std::vector<std::shared_ptr<Mat<Dtype>>> weight;
      layer_[i]->GetWeight(&weight);
      for (const std::shared_ptr<Mat<Dtype>>& w : weight) {
        // Write the number of weights
        uint32_t weight_size = w->Size();
        res += std::fwrite(reinterpret_cast<void*>(&weight_size), 1,
                           sizeof(uint32_t), fp);
        // Write the weights
        res += std::fwrite(reinterpret_cast<void*>(w->Data()), 1,
                           sizeof(Dtype) * weight_size, fp);
      }
    }
    return res;
  }

  /*!
   * Read the graph from disk.
   *
   *  \param[in]  file_path: path to the file
   *
   *  \return     Data size read from the file
   */
  size_t Load(const char* file_path) const {
    if (!file_path || !*file_path) {
      Report(kError, "Invalid file name");
      return 0;
    }
    std::FILE* fp = std::fopen(file_path, "r");
    if (!fp) {
      Report(kError, "Can't open file '%s' for read", file_path);
      return 0;
    }
    size_t size = Read(fp);
    std::fclose(fp);
    if (!size) {
      // Nothing to write
      std::remove(file_path);
      return 0;
    }
    return size;
  }

  /*!
   * Save the graph on disk.
   *
   *  \param[in]  file_path: path to the file
   *
   *  \return     Data size written to the file
   */
  size_t Save(const char* file_path) const {
    if (!file_path || !*file_path) {
      Report(kError, "Invalid file name");
      return 0;
    }
    std::FILE* fp = std::fopen(file_path, "wb");
    if (!fp) {
      Report(kError, "Can't open file '%s' for write", file_path);
      return 0;
    }
    size_t size = Write(fp);
    std::fclose(fp);
    if (!size) {
      // Nothing to write
      std::remove(file_path);
      return 0;
    }
    return size;
  }

  /*!
   * Add a layer to the graph.
   *
   *  \param[in]  layer: layer to add
   *
   *  \return     Layer output activations
   */
  const std::vector<std::shared_ptr<Mat<Dtype>>>& Add(
    const std::shared_ptr<Layer<Dtype>>& layer) {
    layer_.push_back(layer);
    return layer->Output();
  }

  /*!
   * Add a layer to the graph.
   *
   *  \param[in]  layer: layer to add
   *
   *  \return     Layer output activations
   */
  template <class L>
  const std::vector<std::shared_ptr<Mat<Dtype>>>& Add(
    const char*                                     name,
    const std::vector<std::shared_ptr<Mat<Dtype>>>& in,
    const Param*                                    param) {
    const std::shared_ptr<Layer<Dtype>>& layer =
      std::make_shared<L>(name, in, param);
    return Add(layer);
  }

  /*!
   * Set the batch size.
   *
   *  \param[in]  batch_size: batch size
   */
  void SetBatchSize(uint32_t batch_size) {
    for (size_t i = 0; i < layer_.size(); ++i) {
      layer_[i]->SetBatchSize(batch_size);
    }
  }

  /*!
   * Clear the gradients.
   */
  void ZeroGrad() {
    for (size_t i = 0; i < layer_.size(); ++i) {
      layer_[i]->ZeroGrad();
    }
  }

  /*!
   * Forward pass.
   *
   *  \param[in]  state: state
   */
  void Forward(const State& state) {
    for (size_t i = 0; i < layer_.size(); ++i) {
      layer_[i]->Forward(state);
    }
  }

  /*!
   * Backward pass.
   *
   *  \param[in]  state: state
   */
  void Backward(const State& state) {
    size_t offset = layer_.size() - 1;
    for (size_t i = 0; i < layer_.size(); ++i) {
      layer_[offset - i]->Backward(state);
    }
  }

  /*!
   * Get the data layer.
   *
   *  \return Data layer
   */
  std::shared_ptr<LayerData<Dtype>> DataLayer() const {
    // Get the first layer of the graph
    // It should be a data layer
    std::shared_ptr<LayerData<Dtype>> data;
    if (layer_.size()) {
      data = std::dynamic_pointer_cast<LayerData<Dtype>>(layer_[0]);
    }
    return data;
  }

  /*!
   * Get the loss layer.
   *
   *  \return Loss layer
   */
  template <class Ltype = LayerLoss<Dtype>>
  std::shared_ptr<Ltype> LossLayer() const {
    // Get the last layer of the graph
    // It should be a loss layer
    std::shared_ptr<Ltype> loss_layer;
    if (layer_.size()) {
      loss_layer = std::dynamic_pointer_cast<Ltype>(
        layer_[layer_.size() - 1]);
    }
    return loss_layer;
  }

  /*!
   * Get the loss value.
   *
   *  \return Loss value
   */
  Dtype Loss() const {
    // Get the loss layer and ask for the loss value
    const std::shared_ptr<LayerLoss<Dtype>>& loss_layer = LossLayer();
    if (!loss_layer) {
      return Dtype(0);
    }
    return loss_layer->Loss();
  }

  /*!
   * Model training (forward + backward pass).
   *
   *  \return Loss
   */
  virtual Dtype Train() {
    State state(State::PHASE_TRAIN);
    Forward(state);
    Backward(state);
    return Loss();
  }

  /*!
   * Model testing (inference).
   *
   *  \return Accuracy
   */
  virtual Dtype Test() = 0;
};


}  // namespace jik


#endif  // CORE_MODEL_H_
