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


#ifndef CORE_LOG_H_
#define CORE_LOG_H_


#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <ctime>


namespace jik {


const size_t kLogStrMaxSize    = 0x4000;
const char*  kLogInternalError = "Internal error";


/*!
 * Write a log.
 *
 *  \param[in]  log_file_path: path to the log file
 *  \param[in]  msg          : log message (printf style)
 *
 *  \return     Error?
 */
bool LogMsg(const std::string& log_file_path, const char* msg = nullptr, ...) {
  const size_t kLogFileSizeMax = 0x10000000;

  if (!msg || !*msg) {
    msg = kLogInternalError;
  }

  // Open the file: we open and close this file each time we write
  // a log so if the application crashes, we still get the logs
  std::FILE* log_file = std::fopen(log_file_path.c_str(), "at");
  if (!log_file) {
    return false;
  }

  // Log file size
  std::fseek(log_file, 0, SEEK_END);
  int64_t log_file_size = std::ftell(log_file);
  if (log_file_size < 0) {
    return false;
  }
  std::fseek(log_file, 0, SEEK_SET);

  // If the file is too large, we delete it
  if (log_file_size > int64_t(kLogFileSizeMax)) {
    std::fclose(log_file);
    log_file = std::fopen(log_file_path.c_str(), "wt");
    if (!log_file) {
      return false;
    }
  }

  // Log message
  char fmsg[kLogStrMaxSize];
  std::va_list args;
  va_start(args, msg);
  std::vsnprintf(fmsg, kLogStrMaxSize, msg, args);
  va_end(args);

  // Write the log
  std::fprintf(log_file, "%s\n", fmsg);

  // Close the file
  std::fclose(log_file);

  return true;
}


/*!
 * Write a trace (debug only).
 *
 *  \param[in]  msg: trace message (printf style)
 */
void LogTrace(const char* msg = nullptr, ...) {
#ifdef DEBUG
  const char* kLogDefaultFile = "trace.log";

  // Trace message
  char fmsg[kLogStrMaxSize];
  std::va_list args;
  va_start(args, msg);
  std::vsnprintf(fmsg, kLogStrMaxSize, msg, args);
  va_end(args);
  LogMsg(kLogDefaultFile, fmsg);
#endif  // DEBUG
}


/*!
 * Log levels.
 */
enum LogLevel {
  kInfo = 0,  // Information
  kWarning,   // Warning
  kError      // Error (will terminate the process)
};


/*!
 * Report something.
 *
 *  \param[in]  level: report state
 *  \param[in]  msg  : message (printf style)
 */
void Report(LogLevel level, const char* msg = nullptr, ...) {
  if (!msg || !*msg) {
    msg = kLogInternalError;
  }

  char fmsg[kLogStrMaxSize];
  std::va_list args;
  va_start(args, msg);
  std::vsnprintf(fmsg, kLogStrMaxSize, msg, args);
  va_end(args);

  // State message
  const char* smsg;
  switch (level) {
    // Warning
    case kWarning: {
      smsg = "Warning";
      break;
    }

    // Error
    case kError: {
      smsg = "Error";
      break;
    }

    // Information
    default: {
      smsg = "Info";
      break;
    }
  }

  // Add a timestamp to the message
  const size_t kTimeSize = 0x20;
  char time[kTimeSize];
  std::time_t now;
  std::time(&now);
  std::strftime(time, kTimeSize, "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  std::fprintf(stderr, "[%s @ %s]: %s\n", smsg, time, fmsg);
  LogTrace(fmsg);

  if (level == kError) {
    // In case of error, terminate the process
    std::exit(1);
  }
}


/*!
 * Check something.
 *
 *  \param[in]  level: report state
 *  \param[in]  msg  : message (printf style)
 */
void Check(bool check, const char* msg, ...) {
  if (check) {
    return;
  }

  if (!msg || !*msg) {
    msg = kLogInternalError;
  }

  char fmsg[kLogStrMaxSize];
  std::va_list args;
  va_start(args, msg);
  std::vsnprintf(fmsg, kLogStrMaxSize, msg, args);
  va_end(args);

  Report(kError, fmsg);
}


}  // namespace jik


#endif  // CORE_LOG_H_
