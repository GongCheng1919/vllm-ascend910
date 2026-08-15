#pragma once
// Minimal host-side data utilities for AscendC Kernel Launch tests.
// Provides: arg parsing, RNG, file I/O for binary dumps, PASS/FAIL printers.

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "bf16.h"

// ---- Arg parsing ----
struct Args {
    int rows = 128;
    int cols = 128;
    int k = 1024;
    int group_size = 1024;
    int seed = 1;
    int warmup = 5;
    int repeat = 100;
    bool check = false;
    bool profile = false;
    int stress = 0;            // re-launch + re-check N times (race hunting)
    int blocks = 0;            // 0 => auto (query AI core count)
    bool nosync = false;       // enqueue all repeats, sync once (hides launch overhead)
    bool cold = false;         // rotate over many weight copies to defeat L2 reuse
    bool degenerate = false;   // mid-group only: P0 probe scales (see mg_harness.h)
    std::string load_dir;      // mid-group only: read inputs from here instead of generating
    bool dump = false;
    bool transpose = false;
    float mag = 2.0f;          // input abs range for GenerateBF16 (small => exercise C-path)
    std::string pattern = "random";
    std::string out_dir = "./data";
};

inline Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s(argv[i]);
        auto next_int = [&]() -> int {
            if (i + 1 >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(1); }
            return std::atoi(argv[++i]);
        };
        auto next_str = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << s << "\n"; std::exit(1); }
            return std::string(argv[++i]);
        };
        if (s == "--rows")        a.rows = next_int();
        else if (s == "--cols")   a.cols = next_int();
        else if (s == "--k")      a.k = next_int();
        else if (s == "--group")  a.group_size = next_int();
        else if (s == "--seed")   a.seed = next_int();
        else if (s == "--warmup") a.warmup = next_int();
        else if (s == "--repeat") a.repeat = next_int();
        else if (s == "--check")  a.check = true;
        else if (s == "--stress") a.stress = next_int();
        else if (s == "--blocks") a.blocks = next_int();
        else if (s == "--nosync") a.nosync = true;
        else if (s == "--cold")   a.cold = true;
        else if (s == "--degenerate") a.degenerate = true;
        else if (s == "--load-dir") a.load_dir = next_str();
        else if (s == "--profile") a.profile = true;
        else if (s == "--dump")   a.dump = true;
        else if (s == "--transpose") a.transpose = true;
        else if (s == "--mag")    a.mag = std::atof(argv[++i]);
        else if (s == "--pattern") a.pattern = next_str();
        else if (s == "--out-dir") a.out_dir = next_str();
        else if (s == "-h" || s == "--help") {
            std::cout << "Usage: " << argv[0]
                      << " [--rows R] [--cols C] [--group G] [--seed S]"
                         " [--k K] [--warmup W] [--repeat N] [--pattern P] [--check] [--profile] [--dump] [--out-dir DIR]"
                         " [--degenerate]\n";
            std::exit(0);
        }
        else {
            std::cerr << "unknown arg: " << s << "\n"; std::exit(1);
        }
    }
    return a;
}

// ---- RNG generators ----
inline std::vector<uint16_t> GenerateBF16(size_t n, int seed,
                                          float lo = -2.0f, float hi = 2.0f)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<uint16_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = FP32ToBF16(dist(rng));
    return v;
}

inline std::vector<int8_t> GenerateInt8(size_t n, int seed,
                                        int8_t lo = -16, int8_t hi = 15)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int8_t>(dist(rng));
    return v;
}

// ---- Binary file I/O ----
template <typename T>
inline void WriteBin(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << " for write\n"; std::exit(1); }
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}

template <typename T>
inline std::vector<T> ReadBin(const std::string& path, size_t expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << " for read\n"; std::exit(1); }
    std::vector<T> v(expected_count);
    f.read(reinterpret_cast<char*>(v.data()), expected_count * sizeof(T));
    if (f.gcount() != (std::streamsize)(expected_count * sizeof(T))) {
        std::cerr << "short read on " << path << "\n"; std::exit(1);
    }
    return v;
}

// ---- Comparators ----
struct CompareReport {
    bool pass;
    size_t mismatches;
    double max_abs_diff;
    double mean_abs_diff;
    double snr_db;
    size_t first_bad_idx;
};

inline CompareReport CompareExactInt8(
    const std::vector<int8_t>& got, const std::vector<int8_t>& ref)
{
    CompareReport r{true, 0, 0.0, 0.0, 0.0, SIZE_MAX};
    if (got.size() != ref.size()) {
        std::cerr << "size mismatch: got=" << got.size() << " ref=" << ref.size() << "\n";
        r.pass = false; return r;
    }
    double sum_abs = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        int d = int(got[i]) - int(ref[i]);
        if (d != 0) {
            if (r.first_bad_idx == SIZE_MAX) r.first_bad_idx = i;
            r.mismatches++;
        }
        double ad = std::abs(double(d));
        sum_abs += ad;
        if (ad > max_abs) max_abs = ad;
    }
    r.max_abs_diff = max_abs;
    r.mean_abs_diff = sum_abs / got.size();
    r.pass = (r.mismatches == 0);
    return r;
}

inline CompareReport CompareBF16(
    const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
    float atol = 1e-3f, float rtol = 1e-2f)
{
    CompareReport r{true, 0, 0.0, 0.0, 0.0, SIZE_MAX};
    if (got.size() != ref.size()) {
        std::cerr << "size mismatch\n";
        r.pass = false; return r;
    }
    double sum_abs = 0.0, max_abs = 0.0;
    double sum_sig_sq = 0.0, sum_err_sq = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = BF16ToFP32(got[i]);
        float f = BF16ToFP32(ref[i]);
        double d = double(g) - double(f);
        double ad = std::abs(d);
        double tol = atol + rtol * std::abs(double(f));
        if (ad > tol) {
            if (r.first_bad_idx == SIZE_MAX) r.first_bad_idx = i;
            r.mismatches++;
        }
        sum_abs += ad;
        if (ad > max_abs) max_abs = ad;
        sum_sig_sq += double(f) * double(f);
        sum_err_sq += d * d;
    }
    r.max_abs_diff = max_abs;
    r.mean_abs_diff = sum_abs / got.size();
    r.snr_db = (sum_err_sq > 0.0)
        ? 10.0 * std::log10(sum_sig_sq / sum_err_sq)
        : 999.0;
    r.pass = (r.mismatches == 0);
    return r;
}

inline CompareReport CompareFloat(
    const std::vector<float>& got, const std::vector<float>& ref,
    float atol = 1e-5f, float rtol = 1e-5f)
{
    CompareReport r{true, 0, 0.0, 0.0, 0.0, SIZE_MAX};
    if (got.size() != ref.size()) { r.pass = false; return r; }
    double sum_abs = 0.0, max_abs = 0.0, sum_sig = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = double(got[i]) - double(ref[i]);
        double ad = std::abs(d);
        if (ad > atol + rtol * std::abs(double(ref[i]))) {
            if (r.first_bad_idx == SIZE_MAX) r.first_bad_idx = i;
            r.mismatches++;
        }
        sum_abs += ad; if (ad > max_abs) max_abs = ad;
        sum_sig += double(ref[i]) * double(ref[i]); sum_err += d * d;
    }
    r.max_abs_diff = max_abs; r.mean_abs_diff = sum_abs / got.size();
    r.snr_db = (sum_err > 0.0) ? 10.0 * std::log10(sum_sig / sum_err) : 999.0;
    r.pass = (r.mismatches == 0);
    return r;
}

// ---- Pretty printers ----
inline void PrintHeader(const std::string& op, const Args& a) {
    std::cout << "[shape]      op=" << op
              << " rows=" << a.rows << " cols=" << a.cols
              << " k=" << a.k << " group=" << a.group_size << " seed=" << a.seed << "\n";
}

inline void PrintCheck(const std::string& tensor_name, const CompareReport& r) {
    std::cout << "[check]      " << tensor_name
              << "  " << (r.pass ? "PASS" : "FAIL")
              << "  mismatches=" << r.mismatches
              << "  max_abs=" << std::scientific << std::setprecision(3) << r.max_abs_diff
              << "  mean_abs=" << r.mean_abs_diff;
    if (r.snr_db < 999.0) std::cout << "  snr_db=" << std::fixed << std::setprecision(2) << r.snr_db;
    if (!r.pass && r.first_bad_idx != SIZE_MAX) std::cout << "  first_bad=" << r.first_bad_idx;
    std::cout << "\n";
}

inline void PrintTime(double avg_us, double bw_gbs = -1.0,
                      double tflops = -1.0)
{
    std::cout << "[time]       avg " << std::fixed << std::setprecision(2)
              << avg_us << " us";
    if (bw_gbs > 0)  std::cout << "   bw " << std::fixed << std::setprecision(2) << bw_gbs  << " GB/s";
    if (tflops > 0)  std::cout << "   " << std::fixed << std::setprecision(2) << tflops << " TFLOPS";
    std::cout << "\n";
}

inline void PrintStatus(bool pass) {
    std::cout << "[status]     " << (pass ? "PASS" : "FAIL") << "\n";
}

// ---- Latency timer (uses system clock; for accurate NPU profiling use msprof) ----
template <typename Fn>
double BenchmarkUs(Fn&& run_once, int warmup, int repeat) {
    for (int i = 0; i < warmup; ++i) run_once();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return total_us / repeat;
}
