# transcribe-quantize: re-quant a transcribe.cpp GGUF using ggml's
# reference quantizers. Mirrors whisper.cpp's `quantize` and
# llama.cpp's `llama-quantize`. Links libggml directly so it has access
# to ggml_quantize_chunk and the gguf C API; does NOT link against
# libtranscribe (the tool operates on raw GGUF, not via the loader).

add_executable(transcribe-quantize
    main.cpp
)

target_link_libraries(transcribe-quantize
    PRIVATE
        ggml
)

transcribe_apply_warnings(transcribe-quantize)
