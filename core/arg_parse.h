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


#ifndef CORE_ARG_PARSE_H_
#define CORE_ARG_PARSE_H_


#include <string>
#include <algorithm>


namespace jik {


/*!
 *  \class  ArgParse
 *  \brief  Simple arg parser
 */
class ArgParse {
  // Protected attributes
 protected:
  char*const* first_arg;  // First arg
  char*const* last_arg;   // Last arg


  // Public methods
 public:
  /*!
   * Constructor.
   *
   *  \param[in]  argc: number of args
   *  \param[in]  argv: list of args
   */
  ArgParse(int argc, char* argv[]) {
    first_arg = argv;
    last_arg  = argv + argc;
  }

  /*!
   * Destructor.
   */
  ~ArgParse() {}

  /*!
   * Get an arg value (string).
   *
   *  \param[in]  arg: arg name
   *
   *  \return     Arg value
   */
  const char* Arg(const char* arg) const {
    char*const* itr = std::find(first_arg, last_arg, std::string(arg));
    if (itr == last_arg || ++itr == last_arg) {
      return nullptr;
    }
    return *itr;
  }

  /*!
   * Get an arg value (numeric).
   *
   *  \param[in]  arg        : arg name
   *  \param[in]  default_val: default arg value
   *
   *  \param[out] val        : arg value
   *  \return     Arg found?
   */
  template <typename Dtype>
  bool Arg(const char* arg, Dtype* val = nullptr,
           const Dtype* default_val = nullptr) const {
    const char* sval = Arg(arg);
    if (!sval) {
      if (val) {
        if (default_val) {
          *val = *default_val;
        } else {
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
   * Get an arg value (numeric).
   *
   *  \param[in]  arg        : arg name
   *  \param[in]  default_val: default arg value
   *
   *  \param[out] val        : arg value
   *  \return     Arg found?
   */
  template <typename Dtype>
  bool Arg(const char* arg, const Dtype& default_val,
           Dtype* val = nullptr) const {
    return Arg(arg, val, &default_val);
  }

  /*!
   * Check if an arg is set.
   *
   *  \param[in]  arg: arg name
   *
   *  \return     Arg set?
   */
  bool ArgExists(const char* arg) {
    return std::find(first_arg, last_arg, std::string(arg)) != last_arg;
  }
};


}  // namespace jik


#endif  // CORE_ARG_PARSE_H_
