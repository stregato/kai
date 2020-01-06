package org.soft2.kai.tensors.cuda

import org.junit.Before
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cpu.MultiplicationByTensor

class MultiplicationByTensor: MultiplicationByTensor() {

    @Before
    override fun before() {
        Kernel.default = CudaKernel
    }
}