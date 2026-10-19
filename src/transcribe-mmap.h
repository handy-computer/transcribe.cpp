// transcribe-mmap.h - read-only memory map of a model file.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// MappedFile maps a whole file read-only so model weights can be served
// straight from the page cache instead of being copied into anonymous memory.
// File-backed pages are clean and reclaimable: they do not count against the
// Android low-memory killer's anonymous budget or the iOS/macOS footprint, and
// several models loaded from one file share them.
//
//   POSIX:   open + mmap(PROT_READ, MAP_SHARED).
//   Windows: CreateFileW + CreateFileMappingW(PAGE_READONLY) +
//            MapViewOfFile(FILE_MAP_READ).
//
// The mapping is read-only: a write through it crashes. And it reflects the
// file: truncating or rewriting the file in place while it is mapped turns a
// later access into SIGBUS (POSIX) or EXCEPTION_IN_PAGE_ERROR (Windows).
// Replace a model file by writing a new one and renaming it over the old one.
// On Windows the mapped file cannot be opened for writing or truncated until
// the mapping is released.

#pragma once

#include <cstddef>
#include <string>

namespace transcribe {

class MappedFile {
  public:
    MappedFile() = default;

    ~MappedFile() { reset(); }

    MappedFile(const MappedFile &)             = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    MappedFile(MappedFile &&)                  = delete;
    MappedFile & operator=(MappedFile &&)      = delete;

    // True when this build can map files at all.
    static bool supported() noexcept;

    // Map the file at `path` (UTF-8) read-only and start reading it into the
    // page cache in the background (the weights are read either way: mapped
    // tensors fault in from it, copied ones are streamed from it). Replaces
    // any previous mapping. On failure returns false, leaves the object empty
    // and, when `why` is non-null, stores a short reason for the log.
    bool open(const std::string & path, std::string * why);

    // Unmap. Every tensor pointing into the mapping must be dead by now.
    void reset() noexcept;

    // Page-aligned base of the mapping (MapViewOfFile: allocation-granularity
    // aligned). Non-const because ggml's host-pointer buffers take void *;
    // the pages are still read-only.
    void * addr() const noexcept { return addr_; }

    size_t size() const noexcept { return size_; }

  private:
    void * addr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace transcribe
