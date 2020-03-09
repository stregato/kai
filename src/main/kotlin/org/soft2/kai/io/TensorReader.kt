package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor

interface TensorReader {
    fun read(batchSize: Int = 1): Tensor
    val eof: Boolean
}