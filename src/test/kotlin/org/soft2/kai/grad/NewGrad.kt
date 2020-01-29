package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.tensors.Tensor
import kotlin.math.sin

fun grad(vararg a: Any): Gradient {}

class NewGrad {

    @Test
    fun sinGradTest() {

        fun sin(t: Tensor) {
            t.map { sin(it.toDouble()).toFloat() }
        }

        fun cos(t: Tensor) {
            t.map { kotlin.math.cos(it.toDouble()).toFloat() }
        }

        grad(cos) from sin


    }

}