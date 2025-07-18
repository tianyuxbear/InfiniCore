#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/random_sample.h"

#ifdef ENABLE_CPU_API
#include "cpu/random_sample_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/random_sample_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/random_sample_metax.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/random_sample_aclnn.h"
#endif

__C infiniStatus_t
infiniopCreateRandomSampleDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::random_sample::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::random_sample::NAMESPACE::Descriptor **>(desc_ptr), \
            result,                                                                  \
            probs)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
};

__C infiniStatus_t infiniopGetRandomSampleWorkspaceSize(
    infiniopRandomSampleDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                          \
    case CASE: {                                                      \
        using Ptr = const op::random_sample::NAMESPACE::Descriptor *; \
        *size = reinterpret_cast<Ptr>(desc)->minWorkspaceSize();      \
    }                                                                 \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopRandomSample(
    infiniopRandomSampleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                      \
                        result, probs,                                                  \
                        random_val,                                                     \
                        topp, topk, temperature,                                        \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRandomSampleDescriptor(
    infiniopRandomSampleDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::random_sample::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
