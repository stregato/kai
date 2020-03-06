package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.grad.Optimizer
import org.soft2.kai.tensors.random
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.matrix
import org.soft2.kai.tensors.tensor

class Optimizer {


    @Test
    fun xor() {
        val X = tensor(shape(2), 2, floatArrayOf(1f, 0f, 0f, 1f))
        val W1 = random(shape(2, 2)).makeMutable()
        val W2 = random(shape(2, 2)).makeMutable()
        val E = tensor(shape(2), 2, floatArrayOf(0f, 1f, 1f, 0f))


//        var training = Training(X, E)
    }

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

//        optimizer(c)
    }

}