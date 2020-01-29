package org.soft2.kai.grad

import jdk.nashorn.internal.ir.TernaryNode
import org.soft2.kai.tensors.Tensor

class ScalarMulBP(val a: Float): BackPropagation {
    override fun invoke(x: Tensor): Tensor {
        return x * a
    }
}


fun scalarMulBP(val a: Float, x: Tensor) {
    x.gradient = Gradient(x)

    y.
}







