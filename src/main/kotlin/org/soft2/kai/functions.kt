package org.soft2.kai

import org.soft2.kai.tensors.Tensor

abstract class Transformation: (Tensor) -> Tensor {

    abstract override fun invoke(t: Tensor): Tensor

    abstract fun derive(t: Tensor): Tensor
}


class Relu: Transformation() {
    override fun invoke(t: Tensor) =
        t.map {
            if (it > 0f) it else 0f
        }

    override fun derive(t: Tensor) =
        t.map {
            if (it > 0f) 1f else 0f
        }
}

data class Linear(var w: Tensor, var b: Tensor): Transformation() {
    override fun invoke(t: Tensor) = t*w + b

    override fun derive(t: Tensor) = t


}