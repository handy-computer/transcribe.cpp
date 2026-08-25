#import <Foundation/Foundation.h>
#import "transcribe.h"

static void configure_context(struct transcribe_run_params * params) {
    NSString * context = @"Vocabulary: GGUF, ggml, Qwen3-ASR";
    transcribe_run_params_init(params);
    params->context = context.UTF8String;
}

int main(void) {
    struct transcribe_run_params params;
    configure_context(&params);
    return params.context == NULL || TRANSCRIBE_FEATURE_CONTEXT != 7;
}
