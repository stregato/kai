package org.soft2.kai.grad

import org.junit.Before
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cuda.CudaKernel

class CUDAGrad: CPUGradient() {
    @Before
    override fun before() {
        Kernel.default = CudaKernel
    }

}