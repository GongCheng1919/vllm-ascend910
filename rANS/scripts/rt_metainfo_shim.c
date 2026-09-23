/* Let vllm_ascend 0.23.0's libvllm_ascend_kernels.so load on CANN 8.5.0.
 *
 * That wheel was built against a newer CANN runtime and has exactly ONE
 * undefined symbol here, `rtFunctionGetMetaInfoSize`; the other 22 rt and acl
 * imports all resolve. dlopen fails on it, so torch.ops._C_ascend never
 * registers -- and vllm_ascend calls into that namespace from 149 places with
 * essentially no fallback, which is why the engine dies on npu_gemma_rms_norm.
 *
 * Disassembly says the symbol has exactly one caller in the whole library:
 *
 *     vllm_ascend::mla_preprocess_impl(...)
 *
 * i.e. DeepSeek-style Multi-head Latent Attention, which Qwen3.5 does not use.
 * So this stub makes the library loadable without touching any path the model
 * actually runs through.
 *
 * It ABORTS rather than returning a plausible size. A fabricated meta-info
 * size would corrupt MLA silently; a crash with this message cannot be
 * mistaken for a result. If you see it fire, you are running a model that
 * needs MLA and you need a real CANN, not this shim.
 *
 *   gcc -shared -fPIC -O2 -o librt_metainfo_shim.so rt_metainfo_shim.c
 *   LD_PRELOAD=$PWD/librt_metainfo_shim.so python ...
 */
#include <stdio.h>
#include <stdlib.h>

int rtFunctionGetMetaInfoSize(void *a, void *b, void *c, void *d)
{
    (void)a; (void)b; (void)c; (void)d;
    fprintf(stderr,
            "\n[rt_metainfo_shim] rtFunctionGetMetaInfoSize was actually CALLED.\n"
            "  CANN 8.5.0 does not provide it; this stub exists only so that\n"
            "  libvllm_ascend_kernels.so can load. Its one caller is\n"
            "  vllm_ascend::mla_preprocess_impl (DeepSeek MLA), which the model\n"
            "  you are running was not supposed to reach. Aborting instead of\n"
            "  returning a fabricated size.\n\n");
    abort();
}
