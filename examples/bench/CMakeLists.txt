# transcribe-batch-bench: offline batch throughput sweep.
#
# Not part of the public ABI surface; a measurement harness that times
# transcribe_run_batch across a sweep of batch sizes. See batch_bench.cpp.

add_executable(transcribe-batch-bench
    batch_bench.cpp
)

target_link_libraries(transcribe-batch-bench
    PRIVATE
        transcribe
        transcribe-common-example
)

transcribe_apply_warnings(transcribe-batch-bench)
