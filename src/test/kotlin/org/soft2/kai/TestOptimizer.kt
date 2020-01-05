package org.soft2.kai

import org.junit.Test
import org.soft2.kai.grad.Optimizer

class TestOptimizer {

    @Test
    fun simpleOptimizer() {
        val x = tensor("1 2 3").t()
        val W = random(3, 3).mutable()
        val e = tensor("4 5 6").t()

        val y = W*x
        val c = (e - y) * (e - y).t()

        val optimizer = Optimizer()
        optimizer.weights.add(W)

        optimizer(c)
    }

}