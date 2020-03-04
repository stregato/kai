package org.soft2.kai

import org.junit.Test
import org.soft2.kai.grad.Optimizer
import org.soft2.kai.tensors.random
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.matrix

class TestOptimizer {

    @Test
    fun simpleOptimizer() {
        val x = matrix("1 2 3").t()
        val W = random(shape(3, 3))
        val e = matrix("4 5 6").t()

        W.mutable = true


        val y = W*x
        val c = (e - y) * (e - y).t()

        val optimizer = Optimizer()
//        optimizer.weights.add(W)

        optimizer(c)
    }

}