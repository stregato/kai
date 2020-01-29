package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

typealias C1 = (Tensor) -> Tensor


abstract class Gradient (vararg origin: Tensor){
    val origin: Array<out Tensor> = origin
    abstract operator fun invoke(t: Tensor): Tensor
}



typealias BackPropagation = (Tensor) -> Tensor

fun Tensor.trace(x1: Tensor, d1: C1): Tensor {
    this.traces = arrayOf(Tensor.Trace(x1, d1))
    return this
}

fun Tensor.trace(x1: Tensor, d1: C1, x2: Tensor, d2: C1): Tensor {
    this.traces = arrayOf(Tensor.Trace(x1, d1), Tensor.Trace(x2, d2))
    return this
}

fun Tensor.trace(x1: Tensor, d1: C1, x2: Tensor, d2: C1, x3: Tensor, d3: C1): Tensor {
    this.traces = arrayOf(Tensor.Trace(x1, d1), Tensor.Trace(x2, d2), Tensor.Trace(x3, d3))
    return this
}

fun Tensor.backpropagate(x: Tensor) {
    for ((value, diff) in traces) {
        val d = diff(x)
        value.gradient = d
        value.backpropagate(d)
    }
}
