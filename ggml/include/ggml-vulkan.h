#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_vk_get_device_count(void);
GGML_BACKEND_API void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vk_reg(void);

// Hardware identity of a Vulkan device, as reported by the driver. Queried
// from the physical device at registry time, so it is available before
// ggml_backend_dev_init (no logical device, no pipeline compilation). Lets an
// application decide whether a device is worth using before paying for its
// initialization.
//
//   vendor_id, device_id   VkPhysicalDeviceProperties vendorID / deviceID.
//                          For PCI devices the low 16 bits are the PCI
//                          vendor / device id (0x8086 Intel, 0x1002 AMD,
//                          0x10de NVIDIA).
//   driver_id              VkDriverId (VK_KHR_driver_properties); 0 if unknown.
//   shader_core_count      SMs (NVIDIA, VK_NV_shader_sm_builtins), active
//                          compute units (AMD, VK_AMD_shader_core_properties2)
//                          or Xe-cores (Intel, from the backend's device id
//                          table); 0 if unknown. Not comparable across vendors.
struct ggml_vk_device_hw_info {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t driver_id;
    uint32_t shader_core_count;
};

// Fill *info for a device of the Vulkan registry. Returns false and leaves
// *info untouched if dev is not a Vulkan device. Also reachable through
// ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_get_device_hw_info")
// for dynamically loaded backends.
GGML_BACKEND_API bool ggml_backend_vk_get_device_hw_info(ggml_backend_dev_t dev, struct ggml_vk_device_hw_info * info);
typedef bool (*ggml_backend_vk_get_device_hw_info_t)(ggml_backend_dev_t dev, struct ggml_vk_device_hw_info * info);

#ifdef  __cplusplus
}
#endif
