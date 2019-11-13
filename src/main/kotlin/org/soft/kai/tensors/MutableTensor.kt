package org.soft.kai.tensors

class MutableTensor(shape: IntArray, handle: Handle): Tensor(shape, handle) {

    fun update(alpha: Float, A: Tensor) {
        kernel.update(handle, alpha, A.handle)
    }
}