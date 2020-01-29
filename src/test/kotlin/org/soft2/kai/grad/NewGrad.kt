package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.tensors.Tensor
import kotlin.math.sin

typealias C1 = (Tensor) -> Tensor
typealias C2 = (Tensor, Tensor) -> Tensor
typealias CA = (Tensor, Float) -> Tensor

fun grad(d: C1, f: C1): C1 {
    return f
}

//fun trace(t: Tensor, (Tensor) -> Tensor): Tensor {
//
//}

fun trace(t: Tensor, f: (Tensor) -> Tensor): Tensor {

}

class NewGrad {

    @Test
    fun sinGradTest() {
        fun cos(x: Tensor) = x


        fun sin(x: Tensor) {
            x.map { sin(it.toDouble()).toFloat() } .trace (x, ::cos)
        }
    }


    @Test
    fun scaleTest() {

        fun scale(x: Tensor, alpha: Float) {
            x.map { it*alpha }.trace(x) { it*alpha }
        }

    }

}