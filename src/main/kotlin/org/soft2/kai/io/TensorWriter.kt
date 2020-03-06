package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor

interface TensorWriter {
    fun write(t: Tensor)
}