package org.soft2.kai.input

import org.soft2.kai.tensors.Tensor

interface TensorInput {
    fun read(batchSize: Int): Tensor
}