/*
 * This file is part of Tornado: A heterogeneous programming framework: 
 * https://github.com/beehive-lab/tornado
 *
 * Copyright (c) 2013-2018, APT Group, School of Computer Science,
 * The University of Manchester. All rights reserved.
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
 * Authors: James Clarkson
 *
 */
package uk.ac.manchester.tornado.runtime.api;

import uk.ac.manchester.tornado.common.SchedulableTask;
import uk.ac.manchester.tornado.common.TornadoDevice;
import uk.ac.manchester.tornado.common.enums.Access;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task1;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task10;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task15;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task2;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task3;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task4;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task5;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task6;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task7;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task8;
import uk.ac.manchester.tornado.runtime.api.TornadoFunctions.Task9;

public class TaskSchedule {

    private String taskScheduleName;
    private AbstractTaskGraph taskScheduleImpl;

    public TaskSchedule(String name) {
        this.taskScheduleName = name;
        taskScheduleImpl = new TornadoTaskSchedule(name);
    }

    public <T1> TaskSchedule task(String id, Task1<T1> code, T1 arg) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg));
        return this;
    }

    public <T1, T2> TaskSchedule task(String id, Task2<T1, T2> code, T1 arg1, T2 arg2) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2));
        return this;
    }

    public <T1, T2, T3> TaskSchedule task(String id, Task3<T1, T2, T3> code, T1 arg1, T2 arg2, T3 arg3) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3));
        return this;
    }

    public <T1, T2, T3, T4> TaskSchedule task(String id, Task4<T1, T2, T3, T4> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4));
        return this;
    }

    public <T1, T2, T3, T4, T5> TaskSchedule task(String id, Task5<T1, T2, T3, T4, T5> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6> TaskSchedule task(String id, Task6<T1, T2, T3, T4, T5, T6> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6, T7> TaskSchedule task(String id, Task7<T1, T2, T3, T4, T5, T6, T7> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6, arg7));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6, T7, T8> TaskSchedule task(String id, Task8<T1, T2, T3, T4, T5, T6, T7, T8> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6, T7, T8, T9> TaskSchedule task(String id, Task9<T1, T2, T3, T4, T5, T6, T7, T8, T9> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8,
            T9 arg9) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> TaskSchedule task(String id, Task10<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10> code, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7,
            T8 arg8, T9 arg9, T10 arg10) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10));
        return this;
    }

    public <T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> TaskSchedule task(String id, Task15<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15> code, T1 arg1,
            T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12, T13 arg13, T14 arg14, T15 arg15) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, code, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15));
        return this;
    }

    public TaskSchedule prebuiltTask(String id, String entryPoint, String filename, Object[] args, Access[] accesses, TornadoDevice device, int[] dimensions) {
        taskScheduleImpl.addInner(TaskUtils.createTask(taskScheduleImpl.meta(), id, entryPoint, filename, args, accesses, device, dimensions));
        return this;
    }

    public String getTaskScheduleName() {
        return taskScheduleName;
    }

    public TaskSchedule task(SchedulableTask task) {
        taskScheduleImpl.addInner(task);
        return this;
    }

    public TaskSchedule mapAllTo(TornadoDevice device) {
        taskScheduleImpl.setDevice(device);
        return this;
    }

    public TaskSchedule streamIn(Object... objects) {
        taskScheduleImpl.streamInInner(objects);
        return this;
    }

    public TaskSchedule streamOut(Object... objects) {
        taskScheduleImpl.streamOutInner(objects);
        return this;
    }

    public TaskSchedule schedule() {
        taskScheduleImpl.scheduleInner();
        return this;
    }

    public void execute() {
        taskScheduleImpl.schedule().waitOn();
    }

    public void warmup() {
        taskScheduleImpl.warmup();
    }

    public long getReturnValue(String id) {
        return taskScheduleImpl.getReturnValue(id);
    }

    public void dumpEvents() {
        taskScheduleImpl.dumpEvents();
    }

    public void dumpTimes() {
        taskScheduleImpl.dumpTimes();
    }

    public void dumpProfiles() {
        taskScheduleImpl.dumpProfiles();
    }

    public void clearProfiles() {
        taskScheduleImpl.clearProfiles();
    }

    public void syncObjects() {
        taskScheduleImpl.syncObjects();
    }

    public void syncObject(Object object) {
        taskScheduleImpl.syncObject(object);
    }

    public void syncObjects(Object... objects) {
        taskScheduleImpl.syncObjects(objects);
    }

    public SchedulableTask getTask(String id) {
        return taskScheduleImpl.getTask(id);
    }

    public TornadoDevice getDevice() {
        return taskScheduleImpl.getDevice();
    }

    public void setDevice(TornadoDevice device) {
        taskScheduleImpl.setDevice(device);
    }

    public TornadoDevice getDeviceForTask(String id) {
        return taskScheduleImpl.getDeviceForTask(id);
    }

    public void waitOn() {
        taskScheduleImpl.waitOn();
    }
}
