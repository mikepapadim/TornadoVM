/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2024 APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.ptx;

/**
 * NVRTC (NVIDIA Runtime Compilation) module for compiling CUDA C++ source code
 * at runtime using the NVRTC library.
 */
public class NVRTCModule {
    public final byte[] moduleWrapper;
    public final String kernelFunctionName;
    private int maxBlockSize;
    public final String javaName;
    private final String source;

    /**
     * Compile CUDA C++ source code using NVRTC and load the resulting module.
     *
     * @param name               Module name
     * @param cudaSource         CUDA C++ source code as string
     * @param kernelFunctionName Name of the kernel function
     * @param compileOptions     NVRTC compilation options (e.g., "-arch=compute_75", "-O3")
     */
    public NVRTCModule(String name, String cudaSource, String kernelFunctionName, String[] compileOptions) {
        this.source = cudaSource;
        this.kernelFunctionName = kernelFunctionName;
        this.maxBlockSize = -1;
        this.javaName = name;

        // Compile CUDA C++ source to PTX using NVRTC
        byte[] ptxCode = nvrtcCompile(cudaSource, compileOptions);

        if (ptxCode == null || ptxCode.length == 0) {
            System.err.println("[NVRTCModule] Compilation failed for " + name);
            moduleWrapper = new byte[0];
            return;
        }

        // Load the compiled PTX into a CUDA module
        moduleWrapper = cuModuleLoadData(ptxCode);
    }

    /**
     * Compile CUDA C++ source code to PTX using NVRTC.
     *
     * @param source  CUDA C++ source code
     * @param options Compilation options
     * @return Compiled PTX code as byte array, or empty array on failure
     */
    private static native byte[] nvrtcCompile(String source, String[] options);

    /**
     * Get the NVRTC compilation log (for debugging).
     *
     * @return Compilation log as string
     */
    private static native String nvrtcGetLog();

    /**
     * Load a CUDA module from compiled PTX/CUBIN code.
     *
     * @param ptxCode Compiled PTX or CUBIN code
     * @return Module handle wrapped in byte array
     */
    private static native byte[] cuModuleLoadData(byte[] ptxCode);

    /**
     * Unload a CUDA module.
     *
     * @param module Module handle
     * @return Status code
     */
    private static native long cuModuleUnload(byte[] module);

    /**
     * Get the maximum potential block size for optimal occupancy.
     *
     * @param module   Module handle
     * @param funcName Kernel function name
     * @return Maximum block size
     */
    private static native int cuOccupancyMaxPotentialBlockSize(byte[] module, String funcName);

    public int getPotentialBlockSizeMaxOccupancy() {
        if (maxBlockSize < 0) {
            maxBlockSize = cuOccupancyMaxPotentialBlockSize(moduleWrapper, kernelFunctionName);
        }
        return maxBlockSize;
    }

    public String getSource() {
        return source;
    }

    public boolean isCompilationSuccess() {
        return moduleWrapper.length != 0;
    }

    public void unload() {
        cuModuleUnload(moduleWrapper);
    }

    /**
     * Get the last NVRTC compilation log for debugging purposes.
     *
     * @return Compilation log
     */
    public static String getLastCompilationLog() {
        return nvrtcGetLog();
    }
}
