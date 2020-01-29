package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

typealias C1 = (Tensor) -> Tensor
typealias C2 = (Tensor, Tensor) -> Tensor
typealias Cn = (Array<out Tensor>) -> Array<Tensor>


abstract class Gradient (vararg origin: Tensor){
    val origin: Array<out Tensor> = origin
    abstract operator fun invoke(t: Tensor): Tensor
}


typealias BackPropagation = (Tensor) -> Tensor

fun trace(x: Tensor, gradient: Gradient, fn: C1): Tensor {
    gradient.xs = arrayOf(x)
    val y = fn(x)
    y.gradient = gradient
    return y
}

fun trace(a: Tensor, b: Tensor, gradient: Gradient, fn: C2): Tensor {
    gradient.xs = arrayOf(a,b)
    val y = fn(a, b)
    y.gradient = gradient
    return y
}



fun trace( )