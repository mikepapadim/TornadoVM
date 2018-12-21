#!/usr/bin/env python

#
# This file is part of Tornado: A heterogeneous programming framework: 
# https://github.com/beehive-lab/tornado
#
# Copyright (c) 2013-2018, APT Group, School of Computer Science,
# The University of Manchester. All rights reserved.
# DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
#
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details (a copy is included in the LICENSE file that
# accompanied this code).
#
# You should have received a copy of the GNU General Public License version
# 2 along with this work; if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Authors: Juan Fumero
#

import os
import sys
import argparse
import time
import subprocess
import re

## Include here the new test clasess in Tornado
__TEST_THE_WORLD__ = [
	"uk.ac.manchester.tornado.unittests.TestHello",
	"uk.ac.manchester.tornado.unittests.arrays.TestArrays",
	"uk.ac.manchester.tornado.unittests.functional.TestFunctional",
	"uk.ac.manchester.tornado.unittests.vectortypes.TestFloats",
	"uk.ac.manchester.tornado.unittests.vectortypes.TestDoubles",
	"uk.ac.manchester.tornado.unittests.vectortypes.TestInts",
	"uk.ac.manchester.tornado.unittests.vectortypes.TestVectorAllocation",
	"uk.ac.manchester.tornado.unittests.prebuilt.PrebuiltTest",
	"uk.ac.manchester.tornado.unittests.virtualization.TestsVirtualLayer",
	"uk.ac.manchester.tornado.unittests.tasks.TestSingleTaskSingleDevice",
	"uk.ac.manchester.tornado.unittests.tasks.TestMultipleTasksSingleDevice",
	"uk.ac.manchester.tornado.unittests.images.TestImages",
	"uk.ac.manchester.tornado.unittests.branching.TestConditionals",
	"uk.ac.manchester.tornado.unittests.loops.TestLoops",
	"uk.ac.manchester.tornado.unittests.matrices.TestMatrices",
	"uk.ac.manchester.tornado.unittests.images.TestResizeImage",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsFloats",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsDoubles",
	"uk.ac.manchester.tornado.unittests.dynamic.TestDynamic",
]

## Options
__MAIN_TORNADO_TEST_RUNNER__ = "uk.ac.manchester.tornado.unittests.tools.TornadoTestRunner "
__MAIN_TORNADO_JUNIT__ = "org.junit.runner.JUnitCore "
__IGV_OPTIONS__ = "-Dgraal.Dump=*:verbose -Dgraal.PrintGraph=true -Dgraal.PrintCFG=true "
__PRINT_OPENCL_KERNEL__ = "-Dtornado.opencl.source.print=True "
__DEBUG_TORNADO__ = "-Dtornado.debug=True "
__IGNORE_INTEL_PLATFORM__ = "-Dtornado.ignore.platform=Intel "  # Due to a bug when running with optirun
__PRINT_EXECUTION_TIMER__ = "-Dtornado.debug.executionTime=True "

## 
__VERSION__ = "0.3_21032018"

__TORNADO_TESTS_WHITE_LIST__ = [
	"uk.ac.manchester.tornado.unittests.prebuilt.PrebuiltTest#testPrebuild01",

	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testReductionAnnotationCPUSimple",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testReductionAnnotation", 
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testMultiplicationReduction",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testSequentialReduction",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testReduction01",    
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testMapReduce",               
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testThreadSchuler",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testSumInts2",     
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsIntegers#testSumInts3",

	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsFloats#testSumFloats",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsFloats#testMultFloats",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsDoubles#testSumDoubles",
	"uk.ac.manchester.tornado.unittests.reductions.TestReductionsDoubles#testMultdoubles",
        
        "uk.ac.manchester.tornado.unittests.loops.TestLoops#testInnertForEach",
        "uk.ac.manchester.tornado.unittests.loops.TestLoops#testLoopControlFlowBreakNested",
        "uk.ac.manchester.tornado.unittests.loops.TestLoops#testLoopControlFlowBreakNested2",
	]


__TEST_NOT_PASSED__= False

RED   = "\033[1;31m"  
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"

def composeAllOptions(args):
	""" This method concatenates all JVM options that will be passed to 
		the Tornado VM. New options should be concatenated in this method. 
	"""

	verbose = "-Dtornado.unittests.verbose="
	options = verbose

	if (args.verbose):
		options = options + "True "
	else:
		options = options + "False "

	if (args.igv):
		options = options + __IGV_OPTIONS__

	if (args.debugTornado):
		options = options + __DEBUG_TORNADO__

	if (args.printKernel):
		options = options + __PRINT_OPENCL_KERNEL__

	if (args.device != None):
		options = options + args.device

	if (args.printExecution):
		options = options + __PRINT_EXECUTION_TIMER__

	if (args.jvmFlags != None):
		options = options + args.jvmFlags
	
	return options


def runSingleCommand(cmd, args):
	""" Run a command without processing the result of which tests 
		are passed and failed. This method is used to pass a single 
		test quickly in the terminal.
	"""

	cmd = cmd + " " + args.testClass
	cmd = cmd.split(" ")

	start = time.time()
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	end = time.time()

	print out
	print "Total Time (s): " + str(end-start)


def processStats(out, stats):
	""" It updates the hash table `stats` for reporting the total number 
		of methods that were failed and passed
	"""
	
	global __TEST_NOT_PASSED__ 

	pattern = r'Test: class (?P<test_class>[\w\.]+)*\S*$'
	regex = re.compile(pattern)

	statsProcessing = out.splitlines()
	className = ""
	for line in statsProcessing:
		match = regex.search(line)
		if match != None:
			className = match.groups(0)[0]
		
		l = re.sub(r'(  )+', '', line).strip()

		if (l.find("[PASS]") != -1):
			stats["[PASS]"] = stats["[PASS]"] + 1
		elif (l.find("[FAILED]") != -1) :
			stats["[FAILED]"] = stats["[FAILED]"] + 1
			name = l.split(" ")[2]

			# It removes characters for colors
			name = name[5:-4]
		
			if (name.endswith(".")):
				name = name[:-16]

			if (className + "#" + name in __TORNADO_TESTS_WHITE_LIST__):
				print RED + "Test: " + className + "#" + name + " in whiteList." + RESET
			else:
				## set a flag
				__TEST_NOT_PASSED__ = True
	
	return stats


def runCommandWithStats(command, stats):
	""" Run a command and update the stats dictionary """
	command = command.split(" ")
	p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()

	print err
	print out
	
	return processStats(out, stats)


def runTests(args):
	""" Run the tests using the TornadoTestRunner program """	

	options = composeAllOptions(args)

	stats = {"[PASS]" : 0, "[FAILED]": 0}

	## Run test
	cmd = ""
	if (args.useOptirun):
		cmd = "optirun tornado " + __IGNORE_INTEL_PLATFORM__ + options + " " + __MAIN_TORNADO_TEST_RUNNER__ 
	else:
		cmd = "tornado " + options + " " + __MAIN_TORNADO_TEST_RUNNER__ 
	if (args.testClass != None):

		if (args.fast):
			cmd = cmd + " " + args.testClass
			os.system(cmd)
		else:
			runSingleCommand(cmd, args)
	else:
		start = time.time()
		for t in __TEST_THE_WORLD__:
			command = cmd + t

			if (args.fast):
				os.system(command)
			else:
				stats = runCommandWithStats(command, stats)
		
		end = time.time()
		print CYAN

		if (args.fast == False):
			print GREEN
			print "=================================================="
			print BLUE + "              Unit tests report " + GREEN
			print "=================================================="
			print CYAN
			print stats
			coverage = stats["[PASS]"] / float((stats["[PASS]"] + stats["[FAILED]"])) * 100.0
			print "Coverage: " + str(round(coverage, 2))  + "%" 
			print GREEN
			print "=================================================="
			print CYAN

		print "Total Time(s): " + str(end-start)
		print RESET
		

def runWithJUnit(args):
	""" Run the tests using JUNIT """

	cmd = "tornado " + __MAIN_TORNADO_JUNIT__ 

	if (args.testClass != None):
		cmd = cmd + args.testClass
		os.system(cmd)
	else:	
		for t in __TEST_THE_WORLD__:
			command = cmd + t
			os.system(command)


def parseArguments():
	""" Parse command line arguments """ 
	parser = argparse.ArgumentParser(description='Tool to execute tests in Tornado')
	parser.add_argument('testClass', nargs="?", help='testClass#method')
	parser.add_argument('--version', action="store_true", dest="version", default=False, help="Print version")
	parser.add_argument('--verbose', "-V", action="store_true", dest="verbose", default=False, help="Run test in verbose mode")	
	parser.add_argument('--printKernel', "-pk", action="store_true", dest="printKernel", default=False, help="Print OpenCL kernel")	
	parser.add_argument('--junit', action="store_true", dest="junit", default=False, help="Run within JUnitCore main class")	
	parser.add_argument('--igv', action="store_true", dest="igv", default=False, help="Dump GraalIR into IGV")	
	parser.add_argument('--debug', "-d", action="store_true", dest="debugTornado", default=False, help="Debug Tornado")
	parser.add_argument('--fast', "-f", action="store_true", dest="fast", default=False, help="Visualize Fast")	
	parser.add_argument('--optirun', "-optirun", action="store_true", dest="useOptirun", default=False, help="Use optirun with Tornado")	
	parser.add_argument('--device', dest="device", default=None, help="Set an specific device. E.g `s0.t0.device=0:1`")	
	parser.add_argument('--printExec', dest="printExecution", action="store_true", default=False, help="Print OpenCL Kernel Execution Time")	
	parser.add_argument('--jvm', "-J", dest="jvmFlags", required=False, default=None, help="Pass options to the JVM e.g. -J=\"-Ds0.t0.device=0:1\"")	
	args = parser.parse_args()
	return args

def writeStatusInFile():
	f = open(".unittestingStatus", "w")
	if (__TEST_NOT_PASSED__):
		f.write("FAIL")
	else:
		f.write("OK")
	f.close()


def main():
	args = parseArguments()

	if (args.version):
		print __VERSION__
		sys.exit(0)

	if (args.junit):
		runWithJUnit(args)
	else:
		runTests(args)	

	writeStatusInFile()
	if (__TEST_NOT_PASSED__):
		# return error
		sys.exit(1)

if __name__ == '__main__':
	main()

