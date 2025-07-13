#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>

namespace UTIL {

class Timer {
 public:
  Timer(bool start) : start(start) { reset(); }
  void reset() { start_time_ = std::chrono::steady_clock::now(); }
  void update_time() {start_time_ = std::chrono::steady_clock::now();}
  float elapsed_ms()
  {
    auto end_time = std::chrono::steady_clock::now();
    auto dur =
      std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end_time - start_time_);
    return dur.count();
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
  bool start = false;
};

/*
--num
--dim
--topk
--queries
*/


class CommandLineParser {
public:
    CommandLineParser(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            arguments.push_back(argv[i]);
        }
    }

    bool HasOption(const std::string& option) const {
        for (const std::string& arg : arguments) {
            if (arg == option) {
                return true;
            }
        }
        return false;
    }

    std::string GetOptionValue(const std::string& option) const {
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (arguments[i] == option && i + 1 < arguments.size()) {
                return arguments[i + 1];
            }
        }
        return "";
    }

private:
    std::vector<std::string> arguments;
};



class Perf {
public:
  Perf(int n_queries, int batch_size) : n_queries(n_queries), timer(true), batch_size(batch_size) {}

  void setTime(float cur_run_time) {
    this->cur_run_search_times.push_back((cur_run_time) / 1000.0f);
    this->all_run_search_time += (cur_run_time) / 1000.0f;
    this->time_cnt += 1; //one batch map one time_cnt
  }

  std::tuple<float, float, float, float> getPerfResult() {
    this->qps = (this->time_cnt * this->batch_size) / (this->all_run_search_time) ; // this->batch_cnt * this->batch_size = all queries
    std::sort(cur_run_search_times.begin(), cur_run_search_times.end());
    const auto calc_percentile_pos = [](float percentile, size_t N) {
        return static_cast<size_t>(std::ceil(percentile / 100.0 * N)) - 1;
    };
    const float search_time_p99 = cur_run_search_times[calc_percentile_pos(99, cur_run_search_times.size())];
    best_search_time_p99        = std::min(best_search_time_p99, search_time_p99);
    //best_search_time_p99 /= this->batch_size;
    this->avg_query_time = (this->all_run_search_time) / (this->time_cnt);
    return std::make_tuple(this->qps, this->all_run_search_time, this->avg_query_time, this->best_search_time_p99);
  }

private:
  std::vector<float> cur_run_search_times;
  float all_run_search_time = 0.0;
  float avg_query_time = 0.0;
  float qps = 0.0;
  int n_queries = 0;
  int batch_size = 0;
  float best_search_time_p99 = std::numeric_limits<float>::max();
  int time_cnt = 0;

public:
  UTIL::Timer timer;

};


} //end UTIL
