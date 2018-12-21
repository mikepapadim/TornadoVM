/*
 * Copyright (c) 2018, APT Group, School of Computer Science,
 * The University of Manchester. All rights reserved.
 * Copyright (c) 2009, 2017, Oracle and/or its affiliates. All rights reserved.
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
package uk.ac.manchester.tornado.runtime.graal.backend;

import org.graalvm.compiler.core.target.Backend;
import org.graalvm.compiler.phases.util.Providers;

import uk.ac.manchester.tornado.runtime.common.RuntimeUtilities;

public abstract class TornadoBackend<P extends Providers> extends Backend {

    public static final long DEFAULT_HEAP_ALLOCATION = RuntimeUtilities.parseSize(System.getProperty("tornado.heap.allocation", "512MB"));
    public final static boolean ENABLE_EXCEPTIONS = Boolean.parseBoolean(System.getProperty("tornado.exceptions", "False"));

    protected TornadoBackend(Providers providers) {
        super(providers);
    }

    public abstract String decodeDeopt(long value);

    @Override
    public Providers getProviders() {
        return super.getProviders();
    }

}
