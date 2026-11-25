// binary_file_unit.cpp - unit test for transcribe::load_common::BinaryFile.
//
// Exercises BinaryFile's contract (missing file, zero-length read, sequential
// and out-of-order offset reads, past-EOF rejection, non-ASCII UTF-8 paths)
// and guards against MSVC's std::ifstream 4095-byte chunking regression on
// Windows by checking process ReadOperationCount during a multi-MiB read.

#include "transcribe-load-common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

std::vector<uint8_t> make_pattern(size_t n) {
    std::vector<uint8_t> buf(n);
    for (size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xFFu);
    }
    return buf;
}

bool write_file(const fs::path & path, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(out);
}

void test_missing_and_closed() {
    transcribe::load_common::BinaryFile bf;
    uint8_t                             b = 0;
    CHECK(!bf.read_at(0, &b, 1));
    CHECK(!bf.open((fs::temp_directory_path() / "transcribe_binary_file_missing_999999.bin").u8string()));
    CHECK(!bf.read_at(0, &b, 1));
}

void test_reads_and_bounds(const fs::path & dir) {
    constexpr size_t           k_size = 4u << 20;  // 4 MiB
    const std::vector<uint8_t> want   = make_pattern(k_size);
    const fs::path             file   = dir / "payload.bin";
    CHECK(write_file(file, want));

    transcribe::load_common::BinaryFile bf;
    CHECK(bf.open(file.u8string()));

    // Zero-length read succeeds without touching dst; non-zero with null dst fails.
    CHECK(bf.read_at(0, nullptr, 0));
    CHECK(!bf.read_at(0, nullptr, 1));

    // Full 4 MiB read in a single call. On Windows, std::ifstream would split
    // this into >1000 4 KiB ReadFile syscalls; BinaryFile must issue O(1).
    std::vector<uint8_t> got(k_size, 0);
#if defined(_WIN32)
    IO_COUNTERS io_before{};
    IO_COUNTERS io_after{};
    CHECK(GetProcessIoCounters(GetCurrentProcess(), &io_before) != 0);
#endif
    CHECK(bf.read_at(0, got.data(), got.size()));
#if defined(_WIN32)
    CHECK(GetProcessIoCounters(GetCurrentProcess(), &io_after) != 0);
    const uint64_t read_ops = io_after.ReadOperationCount - io_before.ReadOperationCount;
    if (read_ops == 0 || read_ops > 8) {
        std::fprintf(stderr, "FAIL: 4 MiB BinaryFile::read_at took %llu ReadOps (expected 1..8)\n",
                     static_cast<unsigned long long>(read_ops));
        ++g_failures;
    }
#endif
    CHECK(got == want);

    // Out-of-order and back-to-back sequential reads.
    std::vector<uint8_t> slice(8192);
    CHECK(bf.read_at(12345, slice.data(), slice.size()));
    CHECK(std::memcmp(slice.data(), want.data() + 12345, slice.size()) == 0);

    CHECK(bf.read_at(12345 + slice.size(), slice.data(), slice.size()));
    CHECK(std::memcmp(slice.data(), want.data() + 12345 + slice.size(), slice.size()) == 0);

    CHECK(bf.read_at(0, slice.data(), 256));
    CHECK(std::memcmp(slice.data(), want.data(), 256) == 0);

    // Past-EOF / overflow / short reads fail cleanly, and subsequent valid reads recover.
    CHECK(!bf.read_at(k_size - 16, slice.data(), 32));
    CHECK(!bf.read_at(k_size + 100, slice.data(), 16));
    CHECK(!bf.read_at(UINT64_MAX, slice.data(), 16));
    CHECK(bf.read_at(k_size - 16, slice.data(), 16));
    CHECK(std::memcmp(slice.data(), want.data() + k_size - 16, 16) == 0);

    bf.close();
    CHECK(!bf.read_at(0, slice.data(), 1));
}

void test_utf8_path(const fs::path & dir) {
    // Non-ASCII UTF-8 bytes ("cafe" with e-acute + CJK) escaped so source stays ASCII.
    const fs::path  u8dir = dir / fs::u8path("bf-caf\xC3\xA9-\xE6\x97\xA5\xE6\x9C\xAC");
    std::error_code ec;
    fs::create_directories(u8dir, ec);
    CHECK(!ec);

    const std::vector<uint8_t> want = make_pattern(65536);
    const fs::path             file = u8dir / "weights.bin";
    CHECK(write_file(file, want));

    transcribe::load_common::BinaryFile bf;
    CHECK(bf.open(file.u8string()));

    std::vector<uint8_t> got(want.size(), 0);
    CHECK(bf.read_at(0, got.data(), got.size()));
    CHECK(got == want);
}

}  // namespace

int main() {
    std::error_code ec;
    const fs::path  dir = fs::temp_directory_path(ec) / "transcribe_binary_file_unit";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    CHECK(!ec);

    test_missing_and_closed();
    test_reads_and_bounds(dir);
    test_utf8_path(dir);

    fs::remove_all(dir, ec);

    if (g_failures != 0) {
        std::fprintf(stderr, "binary_file_unit: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("binary_file_unit: ok\n");
    return EXIT_SUCCESS;
}
