#define _POSIX_C_SOURCE 200809L
#include <vulkan/vulkan.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { BINDINGS = 6 };

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize size;
} HostBuffer;

static void vk_ok(VkResult result, const char *what) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "q8_driver: %s failed: VkResult %d\n", what, (int)result);
        exit(2);
    }
}

static void *read_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL || fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "q8_driver: cannot open %s: %s\n", path, strerror(errno));
        exit(2);
    }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "q8_driver: cannot size %s\n", path);
        exit(2);
    }
    void *data = malloc((size_t)n == 0 ? 1 : (size_t)n);
    if (data == NULL || fread(data, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "q8_driver: cannot read %s\n", path);
        exit(2);
    }
    fclose(f);
    *size_out = (size_t)n;
    return data;
}

static uint32_t u32_arg(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v == 0 || v > UINT32_MAX) {
        fprintf(stderr, "q8_driver: invalid %s: %s\n", name, s);
        exit(2);
    }
    return (uint32_t)v;
}

static size_t checked_mul(size_t a, size_t b, const char *name) {
    if (b != 0 && a > SIZE_MAX / b) {
        fprintf(stderr, "q8_driver: %s size product overflows size_t\n", name);
        exit(2);
    }
    return a * b;
}

static size_t checked_add(size_t a, size_t b, const char *name) {
    if (a > SIZE_MAX - b) {
        fprintf(stderr, "q8_driver: %s size sum overflows size_t\n", name);
        exit(2);
    }
    return a + b;
}

static uint64_t timespec_diff_ns(const struct timespec *start, const struct timespec *end) {
    int64_t s = (int64_t)end->tv_sec - (int64_t)start->tv_sec;
    int64_t ns = (int64_t)end->tv_nsec - (int64_t)start->tv_nsec;
    if (ns < 0) {
        s -= 1;
        ns += 1000000000L;
    }
    if (s < 0 || (s == 0 && ns < 0)) {
        return 0;
    }
    return (uint64_t)s * 1000000000ULL + (uint64_t)ns;
}

static uint32_t memory_type(VkPhysicalDevice physical, uint32_t bits) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((bits & (1u << i)) != 0 && (props.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    fprintf(stderr, "q8_driver: no host-visible coherent Vulkan memory type\n");
    exit(2);
}

static void create_host_buffer(VkPhysicalDevice physical, VkDevice device, HostBuffer *b,
                               VkDeviceSize size, VkBufferUsageFlags usage) {
    b->size = size == 0 ? 4 : size;
    VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = b->size,
                             .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    vk_ok(vkCreateBuffer(device, &ci, NULL, &b->buffer), "vkCreateBuffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b->buffer, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = req.size,
                               .memoryTypeIndex = memory_type(physical, req.memoryTypeBits)};
    vk_ok(vkAllocateMemory(device, &ai, NULL, &b->memory), "vkAllocateMemory");
    vk_ok(vkBindBufferMemory(device, b->buffer, b->memory, 0), "vkBindBufferMemory");
    vk_ok(vkMapMemory(device, b->memory, 0, b->size, 0, &b->mapped), "vkMapMemory");
}

static void destroy_host_buffer(VkDevice device, HostBuffer *b) {
    if (b->mapped != NULL) vkUnmapMemory(device, b->memory);
    if (b->buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, b->buffer, NULL);
    if (b->memory != VK_NULL_HANDLE) vkFreeMemory(device, b->memory, NULL);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s shader.spv M tokens blocks case-input.bin result.bin [--timestamps path] [--reps N] [--bench]\n", argv[0]);
        return 2;
    }
    const char *timestamps_path = NULL;
    uint32_t reps = 1;
    int do_timestamps = 0;

    for (int i = 7; i < argc; ++i) {
        if (strcmp(argv[i], "--timestamps") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "q8_driver: missing argument for --timestamps\n");
                return 2;
            }
            timestamps_path = argv[++i];
            do_timestamps = 1;
        } else if (strcmp(argv[i], "--reps") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "q8_driver: missing argument for --reps\n");
                return 2;
            }
            reps = u32_arg(argv[++i], "reps");
            if (reps == 0 || reps > 10000) {
                fprintf(stderr, "q8_driver: invalid reps %u\n", reps);
                return 2;
            }
            do_timestamps = 1;
        } else if (strcmp(argv[i], "--bench") == 0) {
            do_timestamps = 1;
        } else {
            fprintf(stderr, "q8_driver: unknown argument %s\n", argv[i]);
            return 2;
        }
    }

    const uint32_t m = u32_arg(argv[2], "M");
    const uint32_t tokens = u32_arg(argv[3], "tokens");
    const uint32_t blocks = u32_arg(argv[4], "blocks");
    if (m > 1000000u || tokens > 1000000u || blocks > 1000000u) {
        fprintf(stderr, "q8_driver: dimensions exceed the bounded shader-index domain\n");
        return 2;
    }
    size_t shader_size = 0, input_size = 0;
    void *shader_code = read_file(argv[1], &shader_size);
    void *input_data = read_file(argv[5], &input_size);
    size_t words_w = checked_mul(checked_mul(m, blocks, "W"), 8, "W");
    size_t words_dm = checked_mul(checked_mul(m, blocks, "DM"), 2, "DM");
    size_t words_x = checked_mul(checked_mul(tokens, blocks, "X"), 8, "X");
    size_t words_e = checked_mul(tokens, blocks, "E");
    size_t outputs = checked_mul(tokens, m, "output");
    size_t sizes[BINDINGS] = {checked_mul(words_w, 4, "W bytes"),
                              checked_mul(words_dm, 4, "DM bytes"),
                              checked_mul(words_x, 4, "X bytes"),
                              checked_mul(words_e, 4, "E bytes"),
                              checked_mul(outputs, 8, "output bytes"),
                              checked_mul(outputs, 4, "refusal bytes")};
    if (words_w > (1u << 26) || words_dm > (1u << 26) || words_x > (1u << 26) ||
        words_e > (1u << 26) || outputs > (1u << 26)) {
        fprintf(stderr, "q8_driver: buffer element count exceeds the bounded shader-index domain\n");
        return 2;
    }
    size_t expected_input = checked_add(checked_add(sizes[0], sizes[1], "input bytes"),
        checked_add(sizes[2], sizes[3], "input bytes"), "input bytes");
    if (shader_size == 0 || shader_size % 4 != 0 || input_size != expected_input) {
        fprintf(stderr, "q8_driver: malformed shader or case input size (%zu != %zu)\n",
                input_size, expected_input);
        return 2;
    }

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "apeiron-q8-baseline",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance instance = VK_NULL_HANDLE;
    vk_ok(vkCreateInstance(&ici, NULL, &instance), "vkCreateInstance");
    uint32_t count = 0;
    vk_ok(vkEnumeratePhysicalDevices(instance, &count, NULL), "vkEnumeratePhysicalDevices(count)");
    if (count == 0) {
        fprintf(stderr, "q8_driver: no Vulkan physical device\n");
        return 2;
    }
    VkPhysicalDevice *phys = calloc(count, sizeof(*phys));
    if (phys == NULL) return 2;
    vk_ok(vkEnumeratePhysicalDevices(instance, &count, phys), "vkEnumeratePhysicalDevices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    uint32_t chosen_timestamp_valid_bits = 0;
    float chosen_timestamp_period = 0.0f;
    VkBool32 chosen_timestamp_compute_and_graphics = VK_FALSE;
    char chosen_device_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = {0};
    uint32_t chosen_driver_version = 0;
    uint32_t chosen_api_version = 0;

    for (uint32_t i = 0; i < count && physical == VK_NULL_HANDLE; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys[i], &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
        vkGetPhysicalDeviceFeatures2(phys[i], &f2);
        if (f13.shaderIntegerDotProduct != VK_TRUE || f2.features.shaderInt64 != VK_TRUE) continue;
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &qcount, NULL);
        VkQueueFamilyProperties *qp = calloc(qcount, sizeof(*qp));
        if (qp == NULL) return 2;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &qcount, qp);
        for (uint32_t q = 0; q < qcount; ++q) {
            if ((qp[q].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                physical = phys[i];
                family = q;
                chosen_timestamp_valid_bits = qp[q].timestampValidBits;
                chosen_timestamp_period = props.limits.timestampPeriod;
                chosen_timestamp_compute_and_graphics = props.limits.timestampComputeAndGraphics;
                snprintf(chosen_device_name, sizeof(chosen_device_name), "%s", props.deviceName);
                chosen_driver_version = props.driverVersion;
                chosen_api_version = props.apiVersion;
                break;
            }
        }
        free(qp);
        if (physical != VK_NULL_HANDLE)
            fprintf(stderr, "device=%s driver=%u api=%u.%u.%u int64=1 dotProduct=1\n", props.deviceName,
                    props.driverVersion,
                    VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
                    VK_VERSION_PATCH(props.apiVersion));
    }
    free(phys);
    if (physical == VK_NULL_HANDLE) {
        fprintf(stderr, "q8_driver: no Vulkan 1.3 compute device with shaderInt64 and shaderIntegerDotProduct\n");
        return 2;
    }
    VkPhysicalDeviceProperties chosen_props;
    vkGetPhysicalDeviceProperties(physical, &chosen_props);
    for (uint32_t i = 0; i < BINDINGS; ++i) {
        if (sizes[i] > chosen_props.limits.maxStorageBufferRange) {
            fprintf(stderr, "q8_driver: binding %u size %zu exceeds maxStorageBufferRange %u\n",
                    i, sizes[i], chosen_props.limits.maxStorageBufferRange);
            return 2;
        }
    }
    uint32_t dispatch_x = (m + 15u) / 16u;
    uint32_t dispatch_y = (tokens + 31u) / 32u;
    if (dispatch_x > chosen_props.limits.maxComputeWorkGroupCount[0] ||
        dispatch_y > chosen_props.limits.maxComputeWorkGroupCount[1]) {
        fprintf(stderr, "q8_driver: dispatch grid exceeds device workgroup-count limits\n");
        return 2;
    }

    if (do_timestamps) {
        fprintf(stderr, "timestampPeriod=%f timestampComputeAndGraphics=%u timestampValidBits=%u\n",
                chosen_timestamp_period,
                (unsigned)chosen_timestamp_compute_and_graphics,
                chosen_timestamp_valid_bits);
        if (chosen_timestamp_valid_bits == 0) {
            fprintf(stderr, "q8_driver: invalid timestampValidBits (0), timestamps unsupported\n");
            return 2;
        }
        if (chosen_timestamp_valid_bits > 64) {
            fprintf(stderr, "q8_driver: invalid timestampValidBits (%u > 64)\n", chosen_timestamp_valid_bits);
            return 2;
        }
        if (chosen_timestamp_compute_and_graphics != VK_TRUE) {
            fprintf(stderr, "q8_driver: timestampComputeAndGraphics is false\n");
            return 2;
        }
        if (chosen_timestamp_period <= 0.0f || !isfinite(chosen_timestamp_period)) {
            fprintf(stderr, "q8_driver: invalid timestampPeriod (%f)\n", chosen_timestamp_period);
            return 2;
        }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkPhysicalDeviceVulkan13Features enable13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                                  .shaderIntegerDotProduct = VK_TRUE};
    VkPhysicalDeviceFeatures enable = {.shaderInt64 = VK_TRUE};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = &enable13, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .pEnabledFeatures = &enable};
    VkDevice device = VK_NULL_HANDLE;
    vk_ok(vkCreateDevice(physical, &dci, NULL, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    HostBuffer buffers[BINDINGS] = {0};
    size_t offset = 0;
    for (uint32_t i = 0; i < BINDINGS; ++i) {
        create_host_buffer(physical, device, &buffers[i], sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (i < 4) { memcpy(buffers[i].mapped, (uint8_t *)input_data + offset, sizes[i]); offset += sizes[i]; }
        else memset(buffers[i].mapped, 0xa5, sizes[i]);
    }
    free(input_data);

    VkDescriptorSetLayoutBinding db[BINDINGS];
    for (uint32_t i = 0; i < BINDINGS; ++i)
        db[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = BINDINGS, .pBindings = db};
    VkDescriptorSetLayout dsl;
    vk_ok(vkCreateDescriptorSetLayout(device, &dsci, NULL, &dsl), "vkCreateDescriptorSetLayout");
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 12};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VkPipelineLayout pl;
    vk_ok(vkCreatePipelineLayout(device, &plci, NULL, &pl), "vkCreatePipelineLayout");
    VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size, .pCode = shader_code};
    VkShaderModule sm;
    vk_ok(vkCreateShaderModule(device, &smci, NULL, &sm), "vkCreateShaderModule");
    free(shader_code);
    VkPipelineShaderStageCreateInfo stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"};
    VkComputePipelineCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage, .layout = pl};
    VkPipeline pipeline;
    vk_ok(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline), "vkCreateComputePipelines");

    VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = BINDINGS};
    VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool pool;
    vk_ok(vkCreateDescriptorPool(device, &dpci, NULL, &pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
    VkDescriptorSet set;
    vk_ok(vkAllocateDescriptorSets(device, &dsai, &set), "vkAllocateDescriptorSets");
    VkDescriptorBufferInfo bi[BINDINGS];
    VkWriteDescriptorSet writes[BINDINGS];
    for (uint32_t i = 0; i < BINDINGS; ++i) {
        bi[i] = (VkDescriptorBufferInfo){.buffer = buffers[i].buffer, .offset = 0, .range = buffers[i].size};
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
    }
    vkUpdateDescriptorSets(device, BINDINGS, writes, 0, NULL);

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = family};
    VkCommandPool cp;
    vk_ok(vkCreateCommandPool(device, &cpci, NULL, &cp), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    vk_ok(vkAllocateCommandBuffers(device, &cbai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

    VkQueryPool query_pool = VK_NULL_HANDLE;
    HostBuffer query_buffer = {0};
    uint64_t *rep_t0 = NULL;
    uint64_t *rep_t1 = NULL;
    uint64_t *rep_delta_ticks = NULL;
    double *rep_device_ns = NULL;
    uint64_t *rep_submit_ns = NULL;
    uint64_t *rep_wait_ns = NULL;

    if (do_timestamps) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        vk_ok(vkCreateQueryPool(device, &qpci, NULL, &query_pool), "vkCreateQueryPool");
        create_host_buffer(physical, device, &query_buffer, 2 * sizeof(uint64_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        memset(query_buffer.mapped, 0, query_buffer.size);

        rep_t0 = calloc(reps, sizeof(uint64_t));
        rep_t1 = calloc(reps, sizeof(uint64_t));
        rep_delta_ticks = calloc(reps, sizeof(uint64_t));
        rep_device_ns = calloc(reps, sizeof(double));
        rep_submit_ns = calloc(reps, sizeof(uint64_t));
        rep_wait_ns = calloc(reps, sizeof(uint64_t));
        if (!rep_t0 || !rep_t1 || !rep_delta_ticks || !rep_device_ns || !rep_submit_ns || !rep_wait_ns) {
            fprintf(stderr, "q8_driver: failed to allocate rep timing arrays\n");
            return 2;
        }
    }

    for (uint32_t rep = 0; rep < reps; ++rep) {
        vk_ok(vkResetCommandBuffer(cb, 0), "vkResetCommandBuffer");
        vk_ok(vkBeginCommandBuffer(cb, &cbi), "vkBeginCommandBuffer");
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
        uint32_t pc[3] = {m, tokens, blocks};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);

        if (do_timestamps) {
            vkCmdResetQueryPool(cb, query_pool, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, 0);
        }

        vkCmdDispatch(cb, dispatch_x, dispatch_y, 1);

        if (do_timestamps) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, 1);
            vkCmdCopyQueryPoolResults(cb, query_pool, 0, 2, query_buffer.buffer, 0,
                                      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            VkBufferMemoryBarrier q_barrier = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = query_buffer.buffer,
                .offset = 0,
                .size = query_buffer.size,
            };
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 0, NULL, 1, &q_barrier, 0, NULL);
        }

        VkBufferMemoryBarrier host_read[2] = {
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = buffers[4].buffer, .offset = 0, .size = sizes[4]},
            {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
             .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
             .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
             .buffer = buffers[5].buffer, .offset = 0, .size = sizes[5]},
        };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, NULL, 2, host_read, 0, NULL);
        vk_ok(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

        struct timespec sub_start, sub_end;
        clock_gettime(CLOCK_MONOTONIC, &sub_start);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        vk_ok(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        clock_gettime(CLOCK_MONOTONIC, &sub_end);

        struct timespec wait_start, wait_end;
        clock_gettime(CLOCK_MONOTONIC, &wait_start);
        vk_ok(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        clock_gettime(CLOCK_MONOTONIC, &wait_end);

        if (do_timestamps) {
            uint64_t submit_ns = timespec_diff_ns(&sub_start, &sub_end);
            uint64_t wait_ns = timespec_diff_ns(&wait_start, &wait_end);

            uint64_t raw_ts[2] = {0};
            memcpy(raw_ts, query_buffer.mapped, sizeof(raw_ts));
            uint64_t t0 = raw_ts[0];
            uint64_t t1 = raw_ts[1];

            uint64_t mask = (chosen_timestamp_valid_bits == 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << chosen_timestamp_valid_bits) - 1ULL);
            uint64_t delta_ticks = (t1 - t0) & mask;

            if (delta_ticks == 0) {
                fprintf(stderr, "q8_driver: rejected zero timestamp delta (t0=%llu, t1=%llu)\n",
                        (unsigned long long)t0, (unsigned long long)t1);
                return 2;
            }
            if (delta_ticks > (mask >> 1)) {
                fprintf(stderr, "q8_driver: rejected wrapped timestamp delta (t0=%llu, t1=%llu, delta=%llu)\n",
                        (unsigned long long)t0, (unsigned long long)t1, (unsigned long long)delta_ticks);
                return 2;
            }

            double dev_ns = (double)delta_ticks * (double)chosen_timestamp_period;
            if (dev_ns <= 0.0 || !isfinite(dev_ns)) {
                fprintf(stderr, "q8_driver: rejected non-finite or non-positive device_ns (%f)\n", dev_ns);
                return 2;
            }

            rep_t0[rep] = t0;
            rep_t1[rep] = t1;
            rep_delta_ticks[rep] = delta_ticks;
            rep_device_ns[rep] = dev_ns;
            rep_submit_ns[rep] = submit_ns;
            rep_wait_ns[rep] = wait_ns;
        }
    }

    FILE *out = fopen(argv[6], "wb");
    if (out == NULL || fwrite(buffers[4].mapped, 1, sizes[4], out) != sizes[4] ||
        fwrite(buffers[5].mapped, 1, sizes[5], out) != sizes[5]) {
        fprintf(stderr, "q8_driver: cannot write result %s\n", argv[6]);
        return 2;
    }
    fclose(out);

    if (do_timestamps) {
        if (timestamps_path != NULL) {
            FILE *jf = fopen(timestamps_path, "w");
            if (jf == NULL) {
                fprintf(stderr, "q8_driver: cannot open timestamps file %s: %s\n", timestamps_path, strerror(errno));
                return 2;
            }
            fprintf(jf, "{\n");
            fprintf(jf, "  \"device_name\": \"%s\",\n", chosen_device_name);
            fprintf(jf, "  \"driver_version\": %u,\n", chosen_driver_version);
            fprintf(jf, "  \"api_version\": \"%u.%u.%u\",\n",
                    VK_VERSION_MAJOR(chosen_api_version), VK_VERSION_MINOR(chosen_api_version), VK_VERSION_PATCH(chosen_api_version));
            fprintf(jf, "  \"timestamp_period\": %f,\n", (double)chosen_timestamp_period);
            fprintf(jf, "  \"timestamp_compute_and_graphics\": %s,\n", (chosen_timestamp_compute_and_graphics == VK_TRUE) ? "true" : "false");
            fprintf(jf, "  \"timestamp_valid_bits\": %u,\n", chosen_timestamp_valid_bits);
            fprintf(jf, "  \"reps\": [\n");
            for (uint32_t r = 0; r < reps; ++r) {
                fprintf(jf, "    {\"rep\": %u, \"t0\": %llu, \"t1\": %llu, \"delta_ticks\": %llu, \"device_ns\": %.2f, \"submit_ns\": %llu, \"wait_ns\": %llu}%s\n",
                        r,
                        (unsigned long long)rep_t0[r],
                        (unsigned long long)rep_t1[r],
                        (unsigned long long)rep_delta_ticks[r],
                        rep_device_ns[r],
                        (unsigned long long)rep_submit_ns[r],
                        (unsigned long long)rep_wait_ns[r],
                        (r + 1 < reps) ? "," : "");
            }
            fprintf(jf, "  ]\n");
            fprintf(jf, "}\n");
            fclose(jf);
        }
        printf("TIMING reps=%u device_ns=%.2f submit_ns=%llu wait_ns=%llu\n",
               reps, rep_device_ns[0],
               (unsigned long long)rep_submit_ns[0],
               (unsigned long long)rep_wait_ns[0]);

        free(rep_t0);
        free(rep_t1);
        free(rep_delta_ticks);
        free(rep_device_ns);
        free(rep_submit_ns);
        free(rep_wait_ns);
        destroy_host_buffer(device, &query_buffer);
        vkDestroyQueryPool(device, query_pool, NULL);
    }

    vkDestroyCommandPool(device, cp, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, sm, NULL);
    vkDestroyPipelineLayout(device, pl, NULL);
    vkDestroyDescriptorSetLayout(device, dsl, NULL);
    for (uint32_t i = 0; i < BINDINGS; ++i) destroy_host_buffer(device, &buffers[i]);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
