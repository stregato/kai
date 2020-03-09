package org.soft2.kai.tensors.cuda

import org.junit.Assume
import org.junit.Before
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cpu.MultiplicationByScalar

open class MultiplicationByScalar: MultiplicationByScalar() {

    @Before
    override fun before() {

        Assume.assumeTrue(CudaKernel.available)
        Kernel.default = CudaKernel
    }
}