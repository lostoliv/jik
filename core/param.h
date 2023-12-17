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


#ifndef CORE_PARAM_H_
#define CORE_PARAM_H_


#include <core/log.h>
#include <string>
#include <unordered_map>


namespace jik {


/*!
 *  \class  State
 *  \brief  List of parameters
 */
class Param {
  // Protected attributes
 protected:
  std::unordered_map<std::string, std::string> param_;  // List of parameters
                                                        // (name, value)


  // Public methods
 public:
  /*!
   * Default constructor.
   */
  Param() {}

  /*!
   * Destructor.
   */
  ~Param() {}

  /*!
   * Get a value (string).
   *
   *  \param[in]  name: parameter name
   *
   *  \param[out] val : parameter value
   *  \return     Parameter found?
   */
  bool Get(const char* name, std::string* val) const {
    auto it = param_.find(name);
    if (it == param_.end()) {
      if (val) {
        val->clear();
      }
      return false;
    }
    if (val) {
      *val = it->second;
    }
    return true;
  }

  /*!
   * Get a value (numeric).
   *
   *  \param[in]  name       : parameter name
   *  \param[in]  default_val: default parameter value
   *
   *  \param[out] val        : parameter value
   *  \return     Parameter found?
   */
  template <typename Dtype>
  bool Get(const char* name, Dtype* val,
           const Dtype* default_val = nullptr) const {
    std::string sval;
    if (!Get(name, &sval)) {
      if (val) {
        if (default_val) {
          *val = *default_val;
        } else {
          Report(kError, "No value '%s' found", name);
          *val = Dtype(0);
        }
      }
      return false;
    }
    if (val) {
      *val = Dtype(std::stod(sval));
    }
    return true;
  }

  /*!
   * Get a value (numeric).
   *
   *  \param[in]  name       : parameter name
   *  \param[in]  default_val: default parameter value
   *
   *  \param[out] val        : parameter value
   *  \return     Parameter found?
   */
  template <typename Dtype>
  bool Get(const char* name, const Dtype& default_val, Dtype* val) const {
    return Get(name, val, &default_val);
  }

  /*!
   * Add a parameter (string).
   *
   *  \param[in] name: parameter name
   *  \param[in] val : parameter value
   */
  void Add(const char* name, const char* val) {
    auto it = param_.find(name);
    if (val) {
      if (it == param_.end()) {
        param_[name] = val;
      } else {
        it->second = val;
      }
    } else {
      if (it != param_.end()) {
        param_.erase(it);
      }
    }
  }

  /*!
   * Add a parameter (numeric).
   *
   *  \param[in] name: parameter name
   *  \param[in] val : parameter value
   */
  template <typename Dtype>
  void Add(const char* name, Dtype val) {
    Add(name, std::to_string(val).c_str());
  }

  /*!
   * Delete a parameter.
   *
   *  \param[in] name: parameter name
   */
  void Remove(const char* name) {
    Add(name, static_cast<const char*>(nullptr));
  }
};


}  // namespace jik


#endif  // CORE_PARAM_H_
