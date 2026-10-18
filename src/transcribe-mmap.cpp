// transcribe-mmap.cpp - read-only memory map of a model file.
//
// See transcribe-mmap.h. The mapping follows llama.cpp's llama_mmap
// (MAP_SHARED + PROT_READ; CreateFileMapping + MapViewOfFile). Once mapped,
// the whole file is read in the background: F_RDADVISE on Apple,
// POSIX_FADV_SEQUENTIAL + POSIX_FADV_WILLNEED on Linux/Android,
// PrefetchVirtualMemory on Windows. On a cold page cache that keeps the first
// run as fast as the copying load's sequential read (MADV_WILLNEED was
// measurably slower on macOS). Never MADV_RANDOM: every family reads
// essentially all of its weights each run, so readahead is what keeps a cold
// start fast. No MAP_POPULATE: it blocks the load on the read and fills every
// page-table entry up front, which costs time even on a warm cache.

#include "transcribe-mmap.h"

#include "transcribe-path.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    define TRANSCRIBE_MMAP_WIN32 1
#elif defined(__unix__) || defined(__APPLE__)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#    define TRANSCRIBE_MMAP_POSIX 1
#endif

namespace transcribe {

namespace {

void set_why(std::string * why, const char * what, const char * detail) {
    if (why != nullptr) {
        *why = what;
        if (detail != nullptr && detail[0] != '\0') {
            *why += ": ";
            *why += detail;
        }
    }
}

#if defined(TRANSCRIBE_MMAP_WIN32)
void set_why_win32(std::string * why, const char * what) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "error %lu", static_cast<unsigned long>(GetLastError()));
    set_why(why, what, buf);
}

// PrefetchVirtualMemory is Windows 8+; resolve it at runtime so the library
// still loads on anything older (where prefetching is a no-op).
struct PrefetchRange {
    void * VirtualAddress;
    SIZE_T NumberOfBytes;
};

typedef BOOL(WINAPI * PrefetchFn)(HANDLE, ULONG_PTR, PrefetchRange *, ULONG);

PrefetchFn resolve_prefetch() {
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (k32 == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<PrefetchFn>(reinterpret_cast<void *>(GetProcAddress(k32, "PrefetchVirtualMemory")));
}

// Start reading [addr, addr + size) in the background.
void prefetch_view(void * addr, size_t size) noexcept {
    static const PrefetchFn prefetch_fn = resolve_prefetch();
    if (prefetch_fn == nullptr) {
        return;
    }
    PrefetchRange range{ addr, size };
    (void) prefetch_fn(GetCurrentProcess(), 1, &range, 0);
}
#endif

}  // namespace

bool MappedFile::supported() noexcept {
#if defined(TRANSCRIBE_MMAP_WIN32) || defined(TRANSCRIBE_MMAP_POSIX)
    return true;
#else
    return false;
#endif
}

#if defined(TRANSCRIBE_MMAP_POSIX)

bool MappedFile::open(const std::string & path, std::string * why) {
    reset();

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        set_why(why, "open failed", std::strerror(errno));
        return false;
    }

    struct stat sb{};

    if (::fstat(fd, &sb) != 0) {
        set_why(why, "fstat failed", std::strerror(errno));
        ::close(fd);
        return false;
    }
    if (!S_ISREG(sb.st_mode) || sb.st_size <= 0) {
        set_why(why, "not a non-empty regular file", nullptr);
        ::close(fd);
        return false;
    }
    if (static_cast<uint64_t>(sb.st_size) > static_cast<uint64_t>(SIZE_MAX)) {
        set_why(why, "file larger than the address space", nullptr);
        ::close(fd);
        return false;
    }
    const size_t size = static_cast<size_t>(sb.st_size);

    void * addr = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        set_why(why, "mmap failed", std::strerror(errno));
        ::close(fd);
        return false;
    }

    // Start reading the file in the background. Hints: failure is harmless.
#    if defined(__APPLE__)
    // radvisory::ra_count is an int, so advise in chunks below 2 GiB.
    const off_t chunk = off_t(1) << 30;
    for (off_t off = 0; off < static_cast<off_t>(size); off += chunk) {
        struct radvisory ra{};
        ra.ra_offset = off;
        ra.ra_count  = static_cast<int>(std::min<off_t>(chunk, static_cast<off_t>(size) - off));
        (void) ::fcntl(fd, F_RDADVISE, &ra);
    }
#    elif defined(__linux__)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
#    else
    (void) ::madvise(addr, size, MADV_WILLNEED);
#    endif
    ::close(fd);  // the mapping holds its own reference to the file

    addr_ = addr;
    size_ = size;
    return true;
}

void MappedFile::reset() noexcept {
    if (addr_ != nullptr) {
        (void) ::munmap(addr_, size_);
    }
    addr_ = nullptr;
    size_ = 0;
}

#elif defined(TRANSCRIBE_MMAP_WIN32)

bool MappedFile::open(const std::string & path, std::string * why) {
    reset();

    // FILE_SHARE_DELETE lets a model manager rename the file away while it is
    // loaded; no FILE_SHARE_WRITE, so nobody can rewrite it underneath us.
    const std::filesystem::path p = path_from_utf8(path);
    HANDLE file = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        set_why_win32(why, "CreateFileW failed");
        return false;
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file, &file_size)) {
        set_why_win32(why, "GetFileSizeEx failed");
        CloseHandle(file);
        return false;
    }
    if (file_size.QuadPart <= 0) {
        set_why(why, "empty file", nullptr);
        CloseHandle(file);
        return false;
    }
    if (static_cast<uint64_t>(file_size.QuadPart) > static_cast<uint64_t>(SIZE_MAX)) {
        set_why(why, "file larger than the address space", nullptr);
        CloseHandle(file);
        return false;
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        set_why_win32(why, "CreateFileMappingW failed");
        CloseHandle(file);
        return false;
    }
    // The section keeps the file referenced and the view keeps the section
    // alive, so neither handle is needed once the view exists.
    void * addr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (addr == nullptr) {
        set_why_win32(why, "MapViewOfFile failed");
    }
    CloseHandle(mapping);
    CloseHandle(file);
    if (addr == nullptr) {
        return false;
    }

    addr_ = addr;
    size_ = static_cast<size_t>(file_size.QuadPart);
    prefetch_view(addr_, size_);
    return true;
}

void MappedFile::reset() noexcept {
    if (addr_ != nullptr) {
        (void) UnmapViewOfFile(addr_);
    }
    addr_ = nullptr;
    size_ = 0;
}

#else

bool MappedFile::open(const std::string & path, std::string * why) {
    (void) path;
    reset();
    set_why(why, "memory mapping is not supported on this platform", nullptr);
    return false;
}

void MappedFile::reset() noexcept {
    addr_ = nullptr;
    size_ = 0;
}

#endif

}  // namespace transcribe
