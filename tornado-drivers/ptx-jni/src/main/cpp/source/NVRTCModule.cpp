/*
 * MIT License
 *
 * Copyright (c) 2024, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <jni.h>
#include <cuda.h>
#include <nvrtc.h>

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>

#ifdef __cplusplus
extern "C" {
#endif

// Helper to convert CUmodule to Java byte array (static to avoid conflicts)
static jbyteArray from_module(JNIEnv *env, CUmodule *module) {
    jbyteArray array = env->NewByteArray(sizeof(CUmodule));
    env->SetByteArrayRegion(array, 0, sizeof(CUmodule), static_cast<const jbyte *>((void *) module));
    return array;
}

// Helper to convert Java byte array to CUmodule (static to avoid conflicts)
static void array_to_module(JNIEnv *env, CUmodule *module_ptr, jbyteArray javaWrapper) {
    env->GetByteArrayRegion(javaWrapper, 0, sizeof(CUmodule), static_cast<jbyte *>((void *) module_ptr));
}

// Global variable to store the last compilation log
static std::string lastCompilationLog;

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    nvrtcCompile
 * Signature: (Ljava/lang/String;[Ljava/lang/String;)[B
 */
JNIEXPORT jbyteArray JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_nvrtcCompile
  (JNIEnv *env, jclass clazz, jstring source, jobjectArray options) {

    // Convert Java string to C string
    const char *cuda_source = env->GetStringUTFChars(source, nullptr);
    if (cuda_source == nullptr) {
        std::cerr << "[NVRTC] Failed to get CUDA source string" << std::endl;
        return env->NewByteArray(0);
    }

    // Create NVRTC program
    nvrtcProgram prog;
    nvrtcResult result = nvrtcCreateProgram(
        &prog,              // program handle
        cuda_source,        // CUDA source code
        "kernel.cu",        // program name (for error messages)
        0,                  // number of headers
        nullptr,            // header sources
        nullptr             // header names
    );

    env->ReleaseStringUTFChars(source, cuda_source);

    if (result != NVRTC_SUCCESS) {
        std::cerr << "[NVRTC] Failed to create program: " << nvrtcGetErrorString(result) << std::endl;
        lastCompilationLog = "Failed to create NVRTC program";
        return env->NewByteArray(0);
    }

    // Convert Java String[] to C++ vector
    jsize numOptions = env->GetArrayLength(options);
    std::vector<const char*> opts(numOptions);
    std::vector<jstring> jstrings(numOptions);

    for (jsize i = 0; i < numOptions; i++) {
        jstrings[i] = (jstring)env->GetObjectArrayElement(options, i);
        opts[i] = env->GetStringUTFChars(jstrings[i], nullptr);
    }

    // Compile the program
    result = nvrtcCompileProgram(
        prog,           // program handle
        numOptions,     // number of options
        opts.data()     // options array
    );

    // Release Java strings
    for (jsize i = 0; i < numOptions; i++) {
        env->ReleaseStringUTFChars(jstrings[i], opts[i]);
    }

    // Get compilation log
    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::vector<char> log(log_size);
    nvrtcGetProgramLog(prog, log.data());
    lastCompilationLog = std::string(log.data());

    if (result != NVRTC_SUCCESS) {
        std::cerr << "[NVRTC] Compilation failed: " << nvrtcGetErrorString(result) << std::endl;
        std::cerr << "[NVRTC] Compilation log:" << std::endl << lastCompilationLog << std::endl;
        nvrtcDestroyProgram(&prog);
        return env->NewByteArray(0);
    }

    // Get compiled PTX
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Debug: print entry point name from PTX and save PTX to file
    std::string ptx_str(ptx.data());
    size_t entry_pos = ptx_str.find(".entry");
    if (entry_pos != std::string::npos) {
        size_t end_pos = ptx_str.find('\n', entry_pos);
        std::string entry_line = ptx_str.substr(entry_pos, end_pos - entry_pos);
        std::cout << "[NVRTC] PTX entry point: " << entry_line << std::endl;
    }

    // Save PTX to file for debugging
    std::ofstream ptx_file("/tmp/nvrtc_generated.ptx");
    if (ptx_file.is_open()) {
        ptx_file << ptx_str;
        ptx_file.close();
        std::cout << "[NVRTC] PTX saved to /tmp/nvrtc_generated.ptx" << std::endl;
    }

    // Convert to Java byte array
    jbyteArray result_array = env->NewByteArray(ptx_size);
    env->SetByteArrayRegion(result_array, 0, ptx_size,
                           reinterpret_cast<jbyte*>(ptx.data()));

    nvrtcDestroyProgram(&prog);

    std::cout << "[NVRTC] Compilation successful (" << ptx_size << " bytes of PTX generated)" << std::endl;
    return result_array;
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    nvrtcGetLog
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_nvrtcGetLog
  (JNIEnv *env, jclass clazz) {
    return env->NewStringUTF(lastCompilationLog.c_str());
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleLoadData
 * Signature: ([B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleLoadData
  (JNIEnv *env, jclass clazz, jbyteArray ptxCode) {

    size_t ptx_length = env->GetArrayLength(ptxCode);
#ifdef _WIN32
    char *ptx = new char[ptx_length + 1];
#else
    char ptx[ptx_length + 1];
#endif
    env->GetByteArrayRegion(ptxCode, 0, ptx_length, reinterpret_cast<jbyte *>(ptx));
    ptx[ptx_length] = 0; // Null-terminate

    CUmodule module;
    CUresult result = cuModuleLoadData(&module, ptx);

#ifdef _WIN32
    delete[] ptx;
#endif

    if (result != CUDA_SUCCESS) {
        std::cerr << "[NVRTC] cuModuleLoadData failed: " << result << std::endl;
        return env->NewByteArray(0);
    }

    std::cout << "[NVRTC] Module loaded successfully, address=" << module << std::endl;
    return from_module(env, &module);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuModuleUnload
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuModuleUnload
  (JNIEnv *env, jclass clazz, jbyteArray module) {
    CUmodule cumodule;
    array_to_module(env, &cumodule, module);
    CUresult result = cuModuleUnload(cumodule);
    return static_cast<jlong>(result);
}

/*
 * Class:     uk_ac_manchester_tornado_drivers_ptx_NVRTCModule
 * Method:    cuOccupancyMaxPotentialBlockSize
 * Signature: ([BLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_uk_ac_manchester_tornado_drivers_ptx_NVRTCModule_cuOccupancyMaxPotentialBlockSize
  (JNIEnv *env, jclass clazz, jbyteArray module, jstring funcName) {
    CUmodule cumodule;
    array_to_module(env, &cumodule, module);

    const char *func_name = env->GetStringUTFChars(funcName, nullptr);

    CUfunction function;
    CUresult result = cuModuleGetFunction(&function, cumodule, func_name);
    env->ReleaseStringUTFChars(funcName, func_name);

    if (result != CUDA_SUCCESS) {
        std::cerr << "[NVRTC] Failed to get function: " << result << std::endl;
        return -1;
    }

    int minGridSize, blockSize;
    result = cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, function, nullptr, 0, 0);

    if (result != CUDA_SUCCESS) {
        std::cerr << "[NVRTC] Failed to get occupancy: " << result << std::endl;
        return -1;
    }

    std::cout << "[NVRTC] Occupancy: minGridSize=" << minGridSize << ", blockSize=" << blockSize << std::endl;
    return static_cast<jint>(blockSize);
}

#ifdef __cplusplus
}
#endif
