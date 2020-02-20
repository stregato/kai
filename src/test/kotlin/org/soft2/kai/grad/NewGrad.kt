package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.tensors.Tensor
import kotlin.math.sin

class NewGrad {

    @Test
    fun sinGradTest() {
        fun cos(x: Tensor) = x


        fun sin(x: Tensor) {
            x.map { sin(it.toDouble()).toFloat() } .trace (x, "Sin", ::cos)
        }
    }


    @Test
    fun scaleTest() {

        fun scale(x: Tensor, alpha: Float) {
            x.map { it*alpha }.trace(x, "Scale") { it*alpha }
        }

    }

}