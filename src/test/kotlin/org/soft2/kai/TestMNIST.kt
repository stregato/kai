package org.soft2.kai

import org.junit.Test
import org.soft2.kai.tensors.*

import org.soft2.kai.tools.MNIST
import kotlin.math.sqrt

class TestMNIST {


    private val batchSize = 32
    private val train = MNIST.training(batchSize)
    private val test = MNIST.test(batchSize)
    private val nh = 50
    private val m = train.m * train.n

    private var w1 = random(shape(m, nh)) / sqrt(m.toDouble()).toFloat()
    private var b1 = zeros(shape(nh))
    private var w2 = random(shape(nh, 1)) / sqrt(nh.toDouble()).toFloat()
    private var b2 = zeros(shape(1))

    init {
        print("w1 mean = ${mean(w1)}, std = ${std(w1)}")
        print("w2 mean = ${mean(w1)}, std = ${std(w2)}")
    }

    private fun lin(x: Tensor, w: Tensor, b: Tensor) = w * x +b

    private fun relu(x: Tensor) =
        x.map {if (it > 0f) it else 0f}


    private fun model(xb: Tensor): Tensor {
        val l1 = lin(xb.reshape(m), w1, b1)
        val l2 = relu(l1)
        return lin(l2, w2, b2)
    }

    @Test
    fun train() {

        // w * x + b

        for (epoch in 1..5) {

            for (data in train) {
                val o = model(data.images)
                val error = o - vector(*data.labels)

            }
        }
    }


}

