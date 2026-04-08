/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standalone benchmark for CUPTI timing observer overhead.
// Creates the observer BEFORE any CUDA call so HES can be enabled.
// Sweeps over row counts and reports a table of overhead per size.

#include <cudf/io/cupti_timing_observer.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cuda/iterator>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

char const* stage_name(cudf::io::parquet_pipeline_stage s)
{
  switch (s) {
    case cudf::io::parquet_pipeline_stage::IO_READ: return "IO_READ";
    case cudf::io::parquet_pipeline_stage::DECOMPRESS: return "DECOMPRESS";
    case cudf::io::parquet_pipeline_stage::PREPROCESS_LEVELS: return "PREPROCESS_LEVELS";
    case cudf::io::parquet_pipeline_stage::COMPUTE_PAGE_SIZES: return "COMPUTE_PAGE_SIZES";
    case cudf::io::parquet_pipeline_stage::COMPUTE_STRING_SIZES: return "COMPUTE_STRING_SIZES";
    case cudf::io::parquet_pipeline_stage::DECODE: return "DECODE";
    default: return "UNKNOWN";
  }
}

// decode_kernel_mask values from parquet_gpu.hpp
char const* decode_kernel_name(uint32_t mask)
{
  switch (mask) {
    case (1 << 0): return "GENERAL";
    case (1 << 1): return "STRING";
    case (1 << 2): return "DELTA_BINARY";
    case (1 << 3): return "DELTA_BYTE_ARRAY";
    case (1 << 4): return "DELTA_LENGTH_BA";
    case (1 << 5): return "FIXED_WIDTH_NO_DICT";
    case (1 << 6): return "FIXED_WIDTH_DICT";
    case (1 << 7): return "BYTE_STREAM_SPLIT";
    case (1 << 8): return "BSS_FW_FLAT";
    case (1 << 9): return "BSS_FW_NESTED";
    case (1 << 10): return "FW_NO_DICT_NESTED";
    case (1 << 11): return "FW_DICT_NESTED";
    case (1 << 12): return "FW_DICT_LIST";
    case (1 << 13): return "FW_NO_DICT_LIST";
    case (1 << 14): return "BSS_FW_LIST";
    case (1 << 15): return "BOOLEAN";
    case (1 << 16): return "BOOLEAN_NESTED";
    case (1 << 17): return "BOOLEAN_LIST";
    case (1 << 18): return "STRING_NESTED";
    case (1 << 19): return "STRING_LIST";
    case (1 << 20): return "STRING_DICT";
    case (1 << 21): return "STRING_DICT_NESTED";
    case (1 << 22): return "STRING_DICT_LIST";
    case (1 << 23): return "STRING_STREAM_SPLIT";
    case (1 << 24): return "STRING_SS_NESTED";
    default: return "UNKNOWN";
  }
}

struct bench_result {
  double cold_ms;  // first iteration
  double warm_ms;  // average of remaining iterations
};

bench_result run_reads(cudf::io::source_info const& src,
                       cudf::io::kernel_timing_observer* observer,
                       int num_iters)
{
  // First iteration = cold
  auto tc0 = std::chrono::high_resolution_clock::now();
  {
    cudf::io::parquet_reader_options opts =
      cudf::io::parquet_reader_options::builder(src);
    if (observer) { opts.set_timing_observer(observer); }
    cudf::io::read_parquet(opts);
    cudaDeviceSynchronize();
  }
  auto tc1 = std::chrono::high_resolution_clock::now();
  double cold_ms = std::chrono::duration<double, std::milli>(tc1 - tc0).count();

  // Remaining iterations = warm
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 1; i < num_iters; ++i) {
    cudf::io::parquet_reader_options opts =
      cudf::io::parquet_reader_options::builder(src);
    if (observer) { opts.set_timing_observer(observer); }
    cudf::io::read_parquet(opts);
    cudaDeviceSynchronize();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double warm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (num_iters - 1);

  return {cold_ms, warm_ms};
}

// Build a mixed-type parquet buffer with the given row count.
std::vector<char> build_parquet(int num_rows)
{
  auto col_int32 = cudf::test::fixed_width_column_wrapper<int32_t>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows));
  auto col_int64 = cudf::test::fixed_width_column_wrapper<int64_t>(
    thrust::make_counting_iterator(0L),
    thrust::make_counting_iterator(static_cast<int64_t>(num_rows)));
  auto col_float = cudf::test::fixed_width_column_wrapper<float>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows));
  auto col_double = cudf::test::fixed_width_column_wrapper<double>(
    thrust::make_counting_iterator(0), thrust::make_counting_iterator(num_rows));

  std::vector<std::string> strings(num_rows);
  for (int i = 0; i < num_rows; ++i) {
    strings[i] = "val_" + std::to_string(i % 10000);
  }
  auto col_str = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  auto tbl = cudf::table_view{{col_int32, col_int64, col_float, col_double, col_str}};

  std::vector<char> buf;
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buf}, tbl)
      .compression(cudf::io::compression_type::SNAPPY);
  cudf::io::write_parquet(write_opts);
  return buf;
}

}  // namespace

int main()
{
  constexpr int num_iters = 42;
  constexpr int row_counts[] = {
    10'000, 100'000, 1'000'000, 5'000'000, 10'000'000, 50'000'000};

  // Create observer BEFORE any CUDA call — this allows HES to initialize.
  auto observer = std::make_unique<cudf::io::cupti_timing_observer>();

  printf("Observer active: %s, HES: %s (error code: %d)\n",
         observer->is_active() ? "yes" : "no",
         observer->is_hes_active() ? "yes" : "no",
         observer->hes_error_code());
  printf("Columns: 5 (int32, int64, float, double, string), SNAPPY\n");
  printf("Iterations per size: %d (1 cold + %d warm)\n\n", num_iters, num_iters - 1);

  // Header
  printf("%-12s %8s %10s %10s %10s %8s %8s  |  %s\n",
         "Rows", "Size MB", "Base ms", "CUPTI ms", "OH ms", "OH %", "Cold %",
         "Per-iter overhead breakdown (us)");
  printf("%-12s %8s %10s %10s %10s %8s %8s  |  %s\n",
         "------------", "--------", "----------", "----------",
         "----------", "--------", "--------",
         "---------------------------------------");

  for (int num_rows : row_counts) {
    // Build parquet for this row count
    auto buf = build_parquet(num_rows);
    double size_mb = buf.size() / (1024.0 * 1024.0);

    auto const src = cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buf.data()), buf.size()}};

    // Measure with observer attached
    observer->reset_overhead_counters();
    auto const attached = run_reads(src, observer.get(), num_iters);
    auto const counters = observer->get_overhead_counters();

    // Measure with observer alive but not attached (to drain any buffered state)
    run_reads(src, nullptr, 3);

    // Destroy observer, measure true baseline, recreate observer
    observer.reset();

    auto const baseline = run_reads(src, nullptr, num_iters);

    // Recreate observer (HES won't re-enable since context already exists,
    // but subscriber + activity tracing will work)
    observer = std::make_unique<cudf::io::cupti_timing_observer>();

    double oh_ms  = attached.warm_ms - baseline.warm_ms;
    double oh_pct = (oh_ms / baseline.warm_ms) * 100.0;

    double cold_oh_ms  = attached.cold_ms - baseline.cold_ms;
    double cold_oh_pct = (cold_oh_ms / baseline.cold_ms) * 100.0;

    // Per-iteration averages of self-profiling counters
    double en_us  = counters.enable_cb_ns / (1000.0 * num_iters);
    double cb_us  = counters.callback_ns / (1000.0 * num_iters);
    double fl_us  = counters.flush_ns / (1000.0 * num_iters);
    double pop_us = counters.populate_ns / (1000.0 * num_iters);
    double ncb    = static_cast<double>(counters.num_callbacks) / num_iters;

    double nreq  = static_cast<double>(counters.num_buf_req) / num_iters;
    double ncomp = static_cast<double>(counters.num_buf_comp) / num_iters;
    double nkr   = static_cast<double>(counters.num_kern_records) / num_iters;

    printf("%-12d %8.1f %10.3f %10.3f %10.3f %+7.2f%% %+7.2f%%"
           "  |  en:%6.0f cb:%6.0f fl:%6.0f pop:%6.0f"
           "  #cb:%.0f  buf:%.0f/%.0f  kr:%.0f\n",
           num_rows, size_mb, baseline.warm_ms, attached.warm_ms,
           oh_ms, oh_pct, cold_oh_pct,
           en_us, cb_us, fl_us, pop_us,
           ncb, nreq, ncomp, nkr);
  }

  // Per-stage detail with throughput for the largest size
  {
    auto buf = build_parquet(row_counts[std::size(row_counts) - 1]);
    auto const src = cudf::io::source_info{cudf::host_span<std::byte const>{
      reinterpret_cast<std::byte const*>(buf.data()), buf.size()}};

    cudf::io::parquet_reader_options opts =
      cudf::io::parquet_reader_options::builder(src);
    opts.set_timing_observer(observer.get());
    auto result = cudf::io::read_parquet(opts);

    printf("\nPer-stage detail (%dM rows):\n",
           row_counts[std::size(row_counts) - 1] / 1'000'000);
    printf("%-24s %-20s %10s %10s %10s %12s %12s\n",
           "Stage", "Kernel", "Time us", "In MB", "Out MB", "In GB/s", "Out GB/s");
    printf("%-24s %-20s %10s %10s %10s %12s %12s\n",
           "------------------------", "--------------------", "----------",
           "----------", "----------", "------------", "------------");

    if (result.metadata.pipeline_stats.has_value()) {
      auto const& stages = result.metadata.pipeline_stats->stages;

      // Collect byte stats keyed by (stage_enum, sub_id)
      // For decompress: sub_id = 0. For decode: sub_id = kernel_mask.
      struct byte_info {
        size_t input_bytes  = 0;
        size_t output_bytes = 0;
      };
      std::map<uint64_t, byte_info> bytes_by_key;

      auto make_key = [](cudf::io::parquet_pipeline_stage s, uint32_t sub) -> uint64_t {
        return (static_cast<uint64_t>(s) << 32) | sub;
      };

      for (auto const& s : stages) {
        if (auto const* d = std::get_if<cudf::io::parquet_decompress_stats>(&s)) {
          auto& b = bytes_by_key[make_key(cudf::io::parquet_pipeline_stage::DECOMPRESS, 0)];
          b.input_bytes += d->input_bytes;
          b.output_bytes += d->output_bytes;
        } else if (auto const* d = std::get_if<cudf::io::parquet_preprocess_levels_stats>(&s)) {
          auto& b =
            bytes_by_key[make_key(cudf::io::parquet_pipeline_stage::PREPROCESS_LEVELS, 0)];
          b.input_bytes += d->input_bytes;
        } else if (auto const* d = std::get_if<cudf::io::parquet_compute_page_sizes_stats>(&s)) {
          auto& b =
            bytes_by_key[make_key(cudf::io::parquet_pipeline_stage::COMPUTE_PAGE_SIZES, 0)];
          b.input_bytes += d->input_bytes;
        } else if (auto const* d =
                     std::get_if<cudf::io::parquet_compute_string_sizes_stats>(&s)) {
          auto& b =
            bytes_by_key[make_key(cudf::io::parquet_pipeline_stage::COMPUTE_STRING_SIZES, 0)];
          b.input_bytes += d->input_bytes;
        } else if (auto const* d = std::get_if<cudf::io::parquet_decode_stats>(&s)) {
          auto& b =
            bytes_by_key[make_key(cudf::io::parquet_pipeline_stage::DECODE, d->kernel_mask)];
          b.input_bytes += d->input_bytes;
          b.output_bytes += d->output_bytes;
        }
      }

      // Print timing + throughput
      for (auto const& s : stages) {
        if (auto const* t = std::get_if<cudf::io::parquet_stage_timing>(&s)) {
          auto key = make_key(t->stage, t->sub_stage_id);
          auto it  = bytes_by_key.find(key);

          double time_us = t->duration_ns / 1000.0;
          double time_s  = t->duration_ns / 1e9;

          // For DECODE stages, show the kernel name; for others show "-"
          char const* sub_name = "-";
          if (t->stage == cudf::io::parquet_pipeline_stage::DECODE && t->sub_stage_id) {
            sub_name = decode_kernel_name(t->sub_stage_id);
          }

          if (it != bytes_by_key.end() && time_s > 0) {
            double in_mb   = it->second.input_bytes / (1024.0 * 1024.0);
            double out_mb  = it->second.output_bytes / (1024.0 * 1024.0);
            double in_gbs  = (it->second.input_bytes / 1e9) / time_s;
            double out_gbs = (it->second.output_bytes / 1e9) / time_s;

            printf("%-24s %-20s %10.1f %10.1f %10.1f %11.1f %11.1f\n",
                   stage_name(t->stage), sub_name,
                   time_us, in_mb, out_mb, in_gbs, out_gbs);
          } else {
            printf("%-24s %-20s %10.1f %10s %10s %12s %12s\n",
                   stage_name(t->stage), sub_name,
                   time_us, "-", "-", "-", "-");
          }
        }
      }
    }
  }

  printf("\nHES: %s\n", observer->is_hes_active() ? "yes" : "no");

  return 0;
}
