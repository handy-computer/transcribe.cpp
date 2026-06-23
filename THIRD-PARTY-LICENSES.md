# Third-party licenses

transcribe.cpp is MIT-licensed (see [`LICENSE`](LICENSE)). It vendors and links
the third-party components below, each under its own permissive license. The
authoritative license text for each ships in-tree at the path noted; copies are
reproduced here for convenience.

| Component | License | Vendored at | Pin |
|-----------|---------|-------------|-----|
| ggml      | MIT     | [`ggml/LICENSE`](ggml/LICENSE) | see [`ggml/UPSTREAM`](ggml/UPSTREAM) |
| miniz     | MIT     | [`src/third_party/miniz/LICENSE`](src/third_party/miniz/LICENSE) | see [`src/third_party/miniz/UPSTREAM`](src/third_party/miniz/UPSTREAM) |

Prebuilt artifacts (the Swift xcframework, the native Python wheels, and the
npm platform packages) carry these same texts alongside the binaries — see each
binding's packaging for the bundled `LICENSE.ggml` / `LICENSE.miniz` files.

---

## ggml

The tensor library underlying transcribe.cpp. Vendored under `ggml/`.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## miniz

A single-file deflate/inflate (zlib-subset) codec. transcribe.cpp uses it only
for Whisper's temperature-fallback compression-ratio heuristic; it replaces the
previous system-zlib dependency. Vendored under `src/third_party/miniz/`.

miniz is MIT-licensed. (The comment block at the top of the amalgamated
`miniz.c` / `miniz.h` still reads "public domain" — a stale artifact from the
project's pre-relicensing history; upstream's current and authoritative license
is the MIT text below, also shipped at `src/third_party/miniz/LICENSE`.)

```
Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC

All Rights Reserved.

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
```
