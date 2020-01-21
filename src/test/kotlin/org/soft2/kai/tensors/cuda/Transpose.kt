package org.soft2.kai.tensors.cuda

import org.junit.Before
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cpu.Transpose

class Transpose: Transpose() {

    @Before
    override fun before() {

        assert(CudaKernel.available)
        Kernel.default = CudaKernel
    }
}