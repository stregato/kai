package org.soft.kai

import org.junit.Test
import org.soft.kai.grad.Optimizer

class TestOptimizer {

    @Test
    fun simpleOptimizer() {
        val x = tensor("1 2 3")
        val W = random(3, 3).mutable()
        val e = tensor("4 5 6")

        val y = W*x
        val c = e*e - y*y

        val optimizer = Optimizer()
        optimizer.weights.add(W)

        optimizer(c)
    }

}