package org.soft.kai.grad

import org.soft.kai.tensors.Tensor

typealias C1 = (Tensor) -> Tensor
typealias C2 = (Tensor, Tensor) -> Tensor
typealias Cn = (Array<out Tensor>) -> Array<Tensor>


abstract class Gradient {
    lateinit var xs: Array<out Tensor>

    abstract operator fun invoke(): Array<Tensor>
}

/*
    y = w1*x
    z = w2*y


    dz/dw1 = dz/dy * dy/dw1


 */

abstract class G1: Gradient() {
    abstract fun `∂x`(x: Tensor): Tensor

    override operator fun invoke(): Array<Tensor> {
        assert( xs.size == 1) {
            "Unexpected size ${xs.size} for input of G1 function"
        }

        return arrayOf(`∂x`(xs[0]))
    }
}


abstract class G2: Gradient() {
    abstract fun `∂a`(a: Tensor, b: Tensor): Tensor

    abstract fun `∂b`(a: Tensor, b: Tensor): Tensor

    override operator fun invoke(): Array<Tensor> {
        assert(xs.size == 2) {
            "Unexpected size ${xs.size} for input of G2 function"
        }

        return arrayOf(`∂a`(xs[0], xs[1]), `∂b`(xs[0], xs[1]))
    }
}


fun grad(x: Tensor, gradient: Gradient, fn: C1): Tensor {
    gradient.xs = arrayOf(x)
    val y = fn(x)
    y.gradient = gradient
    return y
}

fun grad(a: Tensor, b: Tensor, gradient: Gradient, fn: C2): Tensor {
    gradient.xs = arrayOf(a,b)
    val y = fn(a, b)
    y.gradient = gradient
    return y
}
