/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/io/cupti_timing_observer.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda.h>
#include <cupti.h>
#include <cupti_runtime_cbid.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace cudf::io {

namespace {

// Check CUPTI result and throw on error, except for expected failures
#define CUPTI_CHECK(call)                                                              \
  do {                                                                                 \
    CUptiResult _status = (call);                                                      \
    CUDF_EXPECTS(_status == CUPTI_SUCCESS, "CUPTI error: " + std::to_string(_status)); \
  } while (0)

// A parquet read launches O(dozens) of kernels; each activity record is ~150 bytes.
// 64 KB is sufficient and avoids excessive buffer churn.
constexpr size_t kBufferSize    = 64 * 1024;
constexpr size_t kBufferPoolCap = 4;  // max pre-allocated buffers

}  // namespace

struct cupti_timing_observer::impl {
  bool active     = false;
  bool hes_active = false;
  CUptiResult hes_error{CUPTI_SUCCESS};
  CUpti_SubscriberHandle subscriber{};

  // --- Correlation tracking via callback API ---
  // stage_begin/stage_end set the "current stage". The subscriber callback fires
  // synchronously on the same thread for every CUDA runtime API call, recording
  // the CUPTI-assigned correlation ID → our stage encoding.
  bool stage_active = false;
  uint64_t current_ext_id{};  // (stage << 32) | sub_stage_id

  // correlation_id → ext_id — built from callback, only accessed from user thread.
  // Flat vector (not a hash map) for cache-friendly append in the hot callback path.
  struct corr_entry {
    uint32_t correlation_id;
    uint64_t ext_id;
  };
  std::vector<corr_entry> corr_to_ext;

  // --- Kernel records from Activity API ---
  // Written from CUPTI worker thread (buffer_completed), read from populate_stats.
  std::mutex mutex;

  struct kernel_record {
    uint32_t correlation_id;
    uint64_t start_ns;
    uint64_t end_ns;
  };

  std::vector<kernel_record> kernel_records;

  // --- Pre-allocated buffer pool to avoid malloc/free per CUPTI buffer ---
  std::mutex buf_mutex;
  std::vector<uint8_t*> buf_pool;

  uint8_t* acquire_buffer()
  {
    std::lock_guard lock(buf_mutex);
    if (!buf_pool.empty()) {
      auto* b = buf_pool.back();
      buf_pool.pop_back();
      return b;
    }
    return static_cast<uint8_t*>(malloc(kBufferSize));
  }

  void release_buffer(uint8_t* b)
  {
    std::lock_guard lock(buf_mutex);
    if (buf_pool.size() < kBufferPoolCap) {
      buf_pool.push_back(b);
    } else {
      free(b);
    }
  }

  // --- Self-profiling counters (nanoseconds) ---
  using clock = std::chrono::steady_clock;
  uint64_t ns_enable_cb{0};   // cuptiEnableCallback calls in stage_begin/end
  uint64_t ns_callback{0};    // subscriber_callback body (when stage_active)
  uint64_t ns_flush{0};       // cuptiActivityFlushAll in populate_stats
  uint64_t ns_populate{0};    // populate_stats total (sort + join + emit)
  uint32_t num_callbacks{0};    // number of callback invocations during stages
  uint32_t num_buf_req{0};      // number of buffer_requested calls
  uint32_t num_buf_comp{0};     // number of buffer_completed calls
  uint32_t num_kern_records{0}; // total kernel records delivered

  // Static instance pointer for activity buffer callbacks
  static impl* s_instance;

  static void CUPTIAPI subscriber_callback(void* userdata,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid,
                                           void const* cbdata)
  {
    auto* self = static_cast<impl*>(userdata);
    if (!self->stage_active) return;

    auto const* info = static_cast<CUpti_CallbackData const*>(cbdata);
    if (info->callbackSite != CUPTI_API_ENTER) return;

    auto t0 = clock::now();
    self->corr_to_ext.push_back({info->correlationId, self->current_ext_id});
    self->ns_callback +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
    ++self->num_callbacks;
  }

  static void CUPTIAPI buffer_requested(uint8_t** buffer,
                                        size_t* size,
                                        size_t* max_num_records)
  {
    *buffer          = s_instance ? s_instance->acquire_buffer() : nullptr;
    *size            = *buffer ? kBufferSize : 0;
    *max_num_records = 0;
    if (s_instance) { ++s_instance->num_buf_req; }
  }

  static void CUPTIAPI buffer_completed(CUcontext ctx,
                                        uint32_t stream_id,
                                        uint8_t* buffer,
                                        size_t size,
                                        size_t valid_size)
  {
    if (!s_instance || !buffer) return;

    // Collect records locally, then take the lock once.
    std::vector<kernel_record> local;
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto* kernel = reinterpret_cast<CUpti_ActivityKernel9*>(record);
        local.push_back({kernel->correlationId, kernel->start, kernel->end});
      }
    }

    ++s_instance->num_buf_comp;
    s_instance->num_kern_records += local.size();

    if (!local.empty()) {
      std::lock_guard lock(s_instance->mutex);
      s_instance->kernel_records.insert(
        s_instance->kernel_records.end(), local.begin(), local.end());
    }

    s_instance->release_buffer(buffer);
  }
};

cupti_timing_observer::impl* cupti_timing_observer::impl::s_instance = nullptr;

cupti_timing_observer::cupti_timing_observer(config cfg) : _pimpl{std::make_unique<impl>()}
{
  // Set static instance for buffer callbacks
  impl::s_instance = _pimpl.get();

  // HES must be enabled after CUDA driver init but before context creation.
  // cuInit() initialises the driver without creating a context.
  if (cfg.use_hes) {
    CUresult drv = cuInit(0);
    if (drv == CUDA_SUCCESS) {
      CUptiResult hes_res = cuptiActivityEnableHWTrace(1);
      _pimpl->hes_active  = (hes_res == CUPTI_SUCCESS);
      _pimpl->hes_error   = hes_res;
    }
  }

  // Try to subscribe. If another subscriber exists, degrade gracefully.
  CUptiResult res = cuptiSubscribe(
    &_pimpl->subscriber, impl::subscriber_callback, _pimpl.get());
  if (res == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    _pimpl->active = false;
    return;
  }
  CUPTI_CHECK(res);

  // Callbacks start disabled — stage_begin/stage_end toggle them on/off to avoid
  // dispatch overhead for kernel launches outside profiled stages.

  // Register activity buffer callbacks and enable kernel activity tracing.
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(impl::buffer_requested, impl::buffer_completed));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

  _pimpl->active = true;
}

cupti_timing_observer::~cupti_timing_observer()
{
  if (_pimpl->active) {
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiUnsubscribe(_pimpl->subscriber);
  }
  if (impl::s_instance == _pimpl.get()) { impl::s_instance = nullptr; }
  for (auto* b : _pimpl->buf_pool) { free(b); }
}

void cupti_timing_observer::stage_begin(parquet_pipeline_stage stage,
                                        uint32_t sub_stage_id,
                                        rmm::cuda_stream_view)
{
  if (!_pimpl->active) return;
  _pimpl->current_ext_id = (static_cast<uint64_t>(stage) << 32) | sub_stage_id;
  _pimpl->stage_active   = true;
  auto t0                = impl::clock::now();
  cuptiEnableCallback(1, _pimpl->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  cuptiEnableCallback(1, _pimpl->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  _pimpl->ns_enable_cb +=
    std::chrono::duration_cast<std::chrono::nanoseconds>(impl::clock::now() - t0).count();
}

void cupti_timing_observer::stage_end(parquet_pipeline_stage stage,
                                      uint32_t sub_stage_id,
                                      rmm::cuda_stream_view)
{
  if (!_pimpl->active) return;
  auto t0 = impl::clock::now();
  cuptiEnableCallback(0, _pimpl->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);
  cuptiEnableCallback(0, _pimpl->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                      CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060);
  _pimpl->ns_enable_cb +=
    std::chrono::duration_cast<std::chrono::nanoseconds>(impl::clock::now() - t0).count();
  _pimpl->stage_active = false;
}

void cupti_timing_observer::populate_stats(parquet_pipeline_stats& stats)
{
  if (!_pimpl->active) return;

  auto t_pop = impl::clock::now();

  // Force-flush CUPTI activity buffers so all kernel records are delivered.
  // GPU work is already synchronized by the reader. The ~1.5ms flush cost is
  // CUPTI's irreducible overhead for delivering activity records.
  auto t_flush = impl::clock::now();
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  _pimpl->ns_flush +=
    std::chrono::duration_cast<std::chrono::nanoseconds>(impl::clock::now() - t_flush).count();

  std::lock_guard lock(_pimpl->mutex);

  // Sort correlation entries by correlation_id for binary search.
  auto& corr = _pimpl->corr_to_ext;
  std::sort(corr.begin(), corr.end(), [](auto const& a, auto const& b) {
    return a.correlation_id < b.correlation_id;
  });

  // Sum kernel durations per ext_id (encodes stage + sub_stage)
  std::map<uint64_t, uint64_t> duration_by_ext;
  for (auto const& kr : _pimpl->kernel_records) {
    // Binary search for the correlation entry
    auto it = std::lower_bound(
      corr.begin(), corr.end(), kr.correlation_id,
      [](auto const& entry, uint32_t id) { return entry.correlation_id < id; });
    if (it != corr.end() && it->correlation_id == kr.correlation_id) {
      duration_by_ext[it->ext_id] += (kr.end_ns - kr.start_ns);
    }
  }

  // Drain consumed records
  _pimpl->kernel_records.clear();
  corr.clear();

  // Emit parquet_stage_timing entries
  for (auto const& [ext_id, total_ns] : duration_by_ext) {
    auto stage     = static_cast<parquet_pipeline_stage>(ext_id >> 32);
    auto sub_stage = static_cast<uint32_t>(ext_id & 0xFFFFFFFF);
    stats.stages.emplace_back(parquet_stage_timing{stage, sub_stage, total_ns});
  }

  _pimpl->ns_populate +=
    std::chrono::duration_cast<std::chrono::nanoseconds>(impl::clock::now() - t_pop).count();
}

bool cupti_timing_observer::is_active() const { return _pimpl->active; }

bool cupti_timing_observer::is_hes_active() const { return _pimpl->hes_active; }

int cupti_timing_observer::hes_error_code() const { return static_cast<int>(_pimpl->hes_error); }

cupti_timing_observer::overhead_counters cupti_timing_observer::get_overhead_counters() const
{
  return {_pimpl->ns_enable_cb,
          _pimpl->ns_callback,
          _pimpl->ns_flush,
          _pimpl->ns_populate,
          _pimpl->num_callbacks,
          _pimpl->num_buf_req,
          _pimpl->num_buf_comp,
          _pimpl->num_kern_records};
}

void cupti_timing_observer::reset_overhead_counters()
{
  _pimpl->ns_enable_cb    = 0;
  _pimpl->ns_callback     = 0;
  _pimpl->ns_flush        = 0;
  _pimpl->ns_populate     = 0;
  _pimpl->num_callbacks   = 0;
  _pimpl->num_buf_req     = 0;
  _pimpl->num_buf_comp    = 0;
  _pimpl->num_kern_records = 0;
}

}  // namespace cudf::io
