/*
 * This file is part of Tornado: A heterogeneous programming framework:
 * https://github.com/beehive-lab/tornadovm
 *
 * Copyright (c) 2020, APT Group, Department of Computer Science,
 * School of Engineering, The University of Manchester. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */
package uk.ac.manchester.tornado.drivers.ptx;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Set;

import uk.ac.manchester.tornado.api.exceptions.TornadoBailoutRuntimeException;
import uk.ac.manchester.tornado.drivers.ptx.graal.PTXInstalledCode;
import uk.ac.manchester.tornado.drivers.ptx.graal.backend.CodeGenMode;
import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.runtime.tasks.meta.TaskDataContext;

public class PTXCodeCache {

    private final PTXDeviceContext deviceContext;
    private final ConcurrentHashMap<String, PTXInstalledCode> cache;
    
    private static final Set<String> SUPPORTED_PTX_JIT_FLAGS =
            Set.of(
                    CUjitOption.CU_JIT_OPTIMIZATION_LEVEL.name(),
                    CUjitOption.CU_JIT_MAX_REGISTERS.name(),
                    CUjitOption.CU_JIT_CACHE_MODE.name(),
                    CUjitOption.CU_JIT_GENERATE_DEBUG_INFO.name(),
                    CUjitOption.CU_JIT_LOG_VERBOSE.name(),
                    CUjitOption.CU_JIT_GENERATE_LINE_INFO.name(),
                    CUjitOption.CU_JIT_TARGET.name()
            );

    PTXCodeCache(PTXDeviceContext deviceContext) {
        this.deviceContext = deviceContext;
        cache = new ConcurrentHashMap<>();
    }

    public PTXInstalledCode installSource(TaskDataContext taskMeta, String name, byte[] targetCode, String resolvedMethodName, boolean debugKernel) {

        if (!cache.containsKey(name)) {
            if (debugKernel) {
                RuntimeUtilities.dumpKernel(targetCode);
            }

            // Detect code generation mode
            CodeGenMode mode = CodeGenMode.fromSystemProperty();
            boolean isCudaMode = mode == CodeGenMode.CUDA;

            String compilerFlags = taskMeta.getCompilerFlags(TornadoVMBackendType.PTX);
            String[] parts = compilerFlags.trim().split("\\s+");

            if (parts.length % 2 != 0) {
                throw new TornadoBailoutRuntimeException(
                        "Malformed compilerFlags string: expected pairs of <flag> <value>. Got: " + compilerFlags
                );
            }

            if (isCudaMode) {
                // CUDA C++ mode: use NVRTC
                return installSourceCUDA(name, targetCode, resolvedMethodName, parts);
            } else {
                // PTX mode: use existing path
                return installSourcePTX(name, targetCode, resolvedMethodName, parts);
            }
        }

        return cache.get(name);
    }

    private PTXInstalledCode installSourcePTX(String name, byte[] targetCode, String resolvedMethodName, String[] parts) {
        int[] jitOptions = new int[parts.length / 2];
        long[] jitValues = new long[parts.length / 2];

        for (int i = 0; i < parts.length; i += 2) {
            String flagName = parts[i];

            if (!SUPPORTED_PTX_JIT_FLAGS.contains(flagName)) {
                throw new TornadoBailoutRuntimeException(
                        "Unsupported PTX JIT compiler flag: " + flagName +
                                ". Supported flags are: " + SUPPORTED_PTX_JIT_FLAGS
                );
            }

            CUjitOption option;
            try {
                option = CUjitOption.valueOf(flagName);
            } catch (IllegalArgumentException e) {
                throw new TornadoBailoutRuntimeException(
                        "Invalid PTX JIT flag name: " + flagName, e
                );
            }

            jitOptions[i / 2] = option.getValue();

            try {
                jitValues[i / 2] = Long.parseLong(parts[i + 1]);
            } catch (NumberFormatException e) {
                throw new TornadoBailoutRuntimeException(
                        "Invalid flag value (must be integer): '" + parts[i + 1] + "'", e
                );
            }
        }

        PTXModule module = new PTXModule(resolvedMethodName, targetCode, name, jitOptions, jitValues);

        if (module.isPTXJITSuccess()) {
            PTXInstalledCode code = new PTXInstalledCode(name, module, deviceContext);
            cache.put(name, code);
            return code;
        } else {
            throw new TornadoBailoutRuntimeException("PTX JIT compilation failed!");
        }
    }

    private PTXInstalledCode installSourceCUDA(String name, byte[] targetCode, String resolvedMethodName, String[] parts) {
        // Convert byte[] to String (CUDA C++ source)
        String cudaSource = new String(targetCode, StandardCharsets.UTF_8);

        // Convert PTX JIT options to NVRTC options
        List<String> nvrtcOptions = convertToNVRTCOptions(parts);

        // Add default architecture if not specified
        if (nvrtcOptions.stream().noneMatch(opt -> opt.startsWith("-arch="))) {
            // Get compute capability from device
            PTXDevice device = deviceContext.getDevice();
            TargetArchitecture targetArch = device.getTargetArchitecture();
            String arch = String.format("-arch=compute_%d%d", targetArch.getMajor(), targetArch.getMinor());
            nvrtcOptions.add(arch);
        }

        // Compile CUDA C++ using NVRTC
        NVRTCModule module = new NVRTCModule(resolvedMethodName, cudaSource, name,
                                            nvrtcOptions.toArray(new String[0]));

        if (module.isCompilationSuccess()) {
            PTXInstalledCode code = new PTXInstalledCode(name, module, deviceContext);
            cache.put(name, code);
            return code;
        } else {
            String log = NVRTCModule.getLastCompilationLog();
            System.err.println("[CUDA Compilation Error]:");
            System.err.println(log);
            throw new TornadoBailoutRuntimeException("NVRTC compilation failed! See log above.");
        }
    }

    /**
     * Convert PTX JIT options to NVRTC compiler options.
     */
    private List<String> convertToNVRTCOptions(String[] parts) {
        List<String> nvrtcOptions = new ArrayList<>();

        // For now, skip most options to ensure basic compilation works
        // TODO: Add back optimization and other flags once basic compilation is working
        for (int i = 0; i < parts.length; i += 2) {
            String flagName = parts[i];
            String flagValue = parts[i + 1];

            // Map only essential PTX JIT options to NVRTC options
            switch (flagName) {
                case "CU_JIT_TARGET":
                    // Convert SM target (e.g., 75) to arch flag
                    nvrtcOptions.add("-arch=compute_" + flagValue);
                    break;
                // Skip other options for now
                default:
                    break;
            }
        }

        return nvrtcOptions;
    }

    PTXInstalledCode getCachedCode(String name) {
        return cache.get(name);
    }

    boolean isCached(String name) {
        return cache.containsKey(name);
    }

    void reset() {
        for (PTXInstalledCode code : cache.values()) {
            code.invalidate();
        }
        cache.clear();
    }
}
