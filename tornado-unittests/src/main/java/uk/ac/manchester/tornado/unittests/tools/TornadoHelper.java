/*
 * Copyright (c) 2013-2020, 2022-2023, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package uk.ac.manchester.tornado.unittests.tools;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.platform.engine.TestExecutionResult;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.TestExecutionListener;
import org.junit.platform.launcher.TestIdentifier;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;

import static org.junit.platform.engine.discovery.DiscoverySelectors.selectClass;
import static org.junit.platform.engine.discovery.DiscoverySelectors.selectMethod;

import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceFP16NotSupported;
import uk.ac.manchester.tornado.api.exceptions.TornadoDeviceFP64NotSupported;
import uk.ac.manchester.tornado.api.exceptions.TornadoNoOpenCLPlatformException;
import uk.ac.manchester.tornado.unittests.common.SPIRVOptNotSupported;
import uk.ac.manchester.tornado.unittests.common.TornadoNotSupported;
import uk.ac.manchester.tornado.unittests.common.TornadoVMMultiDeviceNotSupported;
import uk.ac.manchester.tornado.unittests.common.TornadoVMOpenCLNotSupported;
import uk.ac.manchester.tornado.unittests.common.TornadoVMPTXNotSupported;
import uk.ac.manchester.tornado.unittests.common.TornadoVMSPIRVNotSupported;
import uk.ac.manchester.tornado.unittests.tools.Exceptions.UnsupportedConfigurationException;

public class TornadoHelper {

    public static final boolean OPTIMIZE_LOAD_STORE_SPIRV = Boolean.parseBoolean(System.getProperty("tornado.spirv.loadstore", "False"));

    private static void printResult(int runCount, int failureCount) {
        System.out.printf("Test ran: %s, Failed: %s%n", runCount, failureCount);
    }

    private static void printResult(int success, int failed, int notSupported) {
        System.out.printf("Test ran: %s, Failed: %s, Unsupported: %s%n", (success + failed + notSupported), failed, notSupported);
    }

    private static void printResult(int success, int failed, int notSupported, StringBuilder buffer) {
        buffer.append(String.format("Test ran: %s, Failed: %s, Unsupported: %s%n", (success + failed + notSupported), failed, notSupported));
    }

    static boolean getProperty(String property) {
        if (System.getProperty(property) != null) {
            return System.getProperty(property).toLowerCase().equals("true");
        }
        return false;
    }

    private static Method getMethodForName(Class<?> klass, String nameMethod) {
        Method method = null;
        for (Method m : klass.getMethods()) {
            if (m.getName().equals(nameMethod)) {
                method = m;
            }
        }
        return method;
    }

    /**
     * It returns the list of methods with the {@link @Test} annotation.
     */
    private static TestSuiteCollection getTestMethods(Class<?> klass) {
        Method[] methods = klass.getMethods();
        ArrayList<Method> methodsToTest = new ArrayList<>();
        HashSet<Method> unsupportedMethods = new HashSet<>();
        HashSet<Method> spirvNotSupportedMethods = new HashSet<>();
        for (Method m : methods) {
            Annotation[] annotations = m.getAnnotations();
            boolean testEnabled = false;
            boolean ignoreTest = false;
            for (Annotation a : annotations) {
                if (a instanceof Disabled) {
                    ignoreTest = true;
                } else if (a instanceof Test) {
                    testEnabled = true;
                } else if (a instanceof TornadoNotSupported) {
                    testEnabled = true;
                    unsupportedMethods.add(m);
                }
            }
            if (testEnabled & !ignoreTest) {
                methodsToTest.add(m);
            }
        }
        return new TestSuiteCollection(methodsToTest, unsupportedMethods, spirvNotSupportedMethods);
    }

    private static class TestResult {
        boolean successful;
        Throwable exception;
        String message;
        String trace;
        String description;

        TestResult(boolean successful) {
            this.successful = successful;
        }

        TestResult(boolean successful, Throwable exception, String description) {
            this.successful = successful;
            this.exception = exception;
            this.description = description;
            if (exception != null) {
                this.message = exception.getMessage();
                StringWriter sw = new StringWriter();
                PrintWriter pw = new PrintWriter(sw);
                exception.printStackTrace(pw);
                this.trace = sw.toString();
            }
        }
    }

    private static class TornadoTestExecutionListener implements TestExecutionListener {
        private final Map<String, TestResult> results = new HashMap<>();
        private int testCount = 0;

        @Override
        public void executionStarted(TestIdentifier testIdentifier) {
            if (testIdentifier.isTest()) {
                testCount++;
            }
        }

        @Override
        public void executionFinished(TestIdentifier testIdentifier, TestExecutionResult testExecutionResult) {
            if (testIdentifier.isTest()) {
                String testId = testIdentifier.getUniqueId();
                if (testExecutionResult.getStatus() == TestExecutionResult.Status.SUCCESSFUL) {
                    results.put(testId, new TestResult(true));
                } else if (testExecutionResult.getStatus() == TestExecutionResult.Status.FAILED) {
                    var throwable = testExecutionResult.getThrowable().orElse(null);
                    results.put(testId, new TestResult(false, throwable, testIdentifier.getDisplayName()));
                }
            }
        }

        TestResult getResult(String testId) {
            return results.get(testId);
        }

        int getTestCount() {
            return testCount;
        }
    }

    static void runTestVerbose(String klassName, String methodName) throws ClassNotFoundException {

        Class<?> klass = Class.forName(klassName);
        ArrayList<Method> methodsToTest = new ArrayList<>();
        TestSuiteCollection suite = null;
        if (methodName == null) {
            suite = getTestMethods(klass);
            methodsToTest = suite.methodsToTest;
        } else {
            Method method = TornadoHelper.getMethodForName(klass, methodName);
            methodsToTest.add(method);
        }

        StringBuilder bufferConsole = new StringBuilder();
        StringBuilder bufferFile = new StringBuilder();

        int successCounter = 0;
        int failedCounter = 0;
        int notSupported = 0;

        bufferConsole.append("Test: " + klass);
        bufferFile.append("Test: " + klass);
        if (methodName != null) {
            bufferConsole.append("#" + methodName);
            bufferFile.append("#" + methodName);
        }
        bufferConsole.append("\n");
        bufferFile.append("\n");

        for (Method m : methodsToTest) {
            String message = String.format("%-50s", "\tRunning test: " + ColorsTerminal.BLUE + m.getName() + ColorsTerminal.RESET);
            bufferConsole.append(message);
            bufferFile.append(message);

            if (suite != null && suite.unsupportedMethods.contains(m)) {
                message = String.format("%20s", " ................ " + ColorsTerminal.YELLOW + " [NOT VALID TEST: UNSUPPORTED] " + ColorsTerminal.RESET + "\n");
                bufferConsole.append(message);
                bufferFile.append(message);
                notSupported++;
                continue;
            }

            LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                    .selectors(selectMethod(klass, m))
                    .build();

            Launcher launcher = LauncherFactory.create();
            TornadoTestExecutionListener listener = new TornadoTestExecutionListener();
            launcher.registerTestExecutionListeners(listener);
            launcher.execute(request);

            // Get the result for this specific test
            TestResult result = null;
            for (var entry : listener.results.entrySet()) {
                result = entry.getValue();
                break; // Only one test was executed
            }

            if (result != null && result.successful) {
                message = String.format("%20s", " ................ " + ColorsTerminal.GREEN + " [PASS] " + ColorsTerminal.RESET + "\n");
                bufferConsole.append(message);
                bufferFile.append(message);
                successCounter++;
            } else if (result != null) {
                var exception = result.exception;

                // Check for unsupported configuration exceptions
                if (exception instanceof UnsupportedConfigurationException) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [UNSUPPORTED CONFIGURATION: At least 2 accelerators are required] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoVMPTXNotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [PTX CONFIGURATION UNSUPPORTED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoNoOpenCLPlatformException) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [OPENCL CONFIGURATION UNSUPPORTED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoVMMultiDeviceNotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [[UNSUPPORTED] MULTI-DEVICE CONFIGURATION REQUIRED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoVMOpenCLNotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [OPENCL CONFIGURATION UNSUPPORTED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoVMSPIRVNotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.PURPLE + " [SPIRV CONFIGURATION UNSUPPORTED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof SPIRVOptNotSupported && OPTIMIZE_LOAD_STORE_SPIRV) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.RED + " [SPIRV OPTIMIZATION NOT SUPPORTED] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    failedCounter++;
                    continue;
                }

                if (exception instanceof TornadoDeviceFP64NotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.YELLOW + " [FP64 UNSUPPORTED FOR CURRENT DEVICE] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                if (exception instanceof TornadoDeviceFP16NotSupported) {
                    message = String.format("%20s", " ................ " + ColorsTerminal.YELLOW + " [FP16 UNSUPPORTED FOR CURRENT DEVICE] " + ColorsTerminal.RESET + "\n");
                    bufferConsole.append(message);
                    bufferFile.append(message);
                    notSupported++;
                    continue;
                }

                message = String.format("%20s", " ................ " + ColorsTerminal.RED + " [FAILED] " + ColorsTerminal.RESET + "\n");
                bufferConsole.append(message);
                bufferFile.append(message);
                failedCounter++;

                bufferConsole.append("\t\t\\_[REASON] " + result.message + "\n");
                bufferFile.append("\t\t\\_[REASON] " + result.message + "\n\t" + result.trace + "\n" + result.description + "\n" + result.exception);
            }
        }

        printResult(successCounter, failedCounter, notSupported, bufferConsole);
        printResult(successCounter, failedCounter, notSupported, bufferFile);
        System.out.println(bufferConsole);

        // Print File
        try (BufferedWriter w = new BufferedWriter(new FileWriter("tornado_unittests.log", true))) {
            DateFormat dateFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
            Date date = new Date();
            w.write("\n" + dateFormat.format(date) + "\n");
            w.write(bufferFile.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static void runTestClassAndMethod(String klassName, String methodName) throws ClassNotFoundException {
        Class<?> klass = Class.forName(klassName);
        Method method = getMethodForName(klass, methodName);

        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(selectMethod(klass, method))
                .build();

        Launcher launcher = LauncherFactory.create();
        TornadoTestExecutionListener listener = new TornadoTestExecutionListener();
        launcher.registerTestExecutionListeners(listener);
        launcher.execute(request);

        int failureCount = 0;
        for (var result : listener.results.values()) {
            if (!result.successful) {
                failureCount++;
            }
        }

        printResult(listener.getTestCount(), failureCount);
    }

    static void runTestClass(String klassName) throws ClassNotFoundException {
        Class<?> klass = Class.forName(klassName);

        LauncherDiscoveryRequest request = LauncherDiscoveryRequestBuilder.request()
                .selectors(selectClass(klass))
                .build();

        Launcher launcher = LauncherFactory.create();
        TornadoTestExecutionListener listener = new TornadoTestExecutionListener();
        launcher.registerTestExecutionListeners(listener);
        launcher.execute(request);

        int failureCount = 0;
        for (var result : listener.results.values()) {
            if (!result.successful) {
                failureCount++;
            }
        }

        printResult(listener.getTestCount(), failureCount);
    }

    static class TestSuiteCollection {
        ArrayList<Method> methodsToTest;
        HashSet<Method> unsupportedMethods;

        TestSuiteCollection(ArrayList<Method> methodsToTest, HashSet<Method> unsupportedMethods, HashSet<Method> spirvUnsupportedMethods) {
            this.methodsToTest = methodsToTest;
            this.unsupportedMethods = unsupportedMethods;
        }
    }
}
