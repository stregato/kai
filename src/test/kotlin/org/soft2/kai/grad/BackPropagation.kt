package org.soft2.kai.grad

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensors.*
import org.soft2.kai.tensors.cpu.CpuKernel

class BackPropagation {
    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }

    @Test
    fun simpleBack() {
        val X = matrix("1 1 1")
        val W = random(shape(3, 3)).makeMutable()
        val E = matrix("-1 0 1")
        var i = 0

        while (true) {
            i++
            val Y = X * W
            val gap = ( Y - E ) * 0.1
            if ( norm0(gap) > 0.001 ) {
                Y backpropagate gap
            } else {
                print("Converged in $i iterations. Final gap=$gap, Y=$Y, W=$W")
                break
            }
        }
    }

    @Test
    fun doubleBack() {
        val X = matrix("1 1 1")
        val W1 = random(shape(3, 3)).makeMutable()
        val W2 = random(shape(3, 3)).makeMutable()
        val E = matrix("-1 0 1")
        var i = 0

        while (true) {
            i++
            val Y = X * W1
            val Z = Y * W2
            val gap = ( Z - E ) * 0.1
            if ( norm0(gap) > 0.001 ) {
                Z backpropagate gap
            } else {
                print("Converged in $i iterations. Final gap=$gap, final Z=$Z")
                break
            }
        }
    }

    @Test
    fun xor() {
        val X = tensor(shape(2), 2, floatArrayOf(1f, 0f, 0f, 1f))
        val W1 = random(shape(2, 2)).makeMutable()
        val W2 = random(shape(2, 2)).makeMutable()
        val E = tensor(shape(2), 2, floatArrayOf(0f, 1f, 1f, 0f))
        var i = 0

        while (true) {
            i++
            val Y = W1 * X
            val Z = W2 * Y
            val error = Z - E
            val loss = norm0(error)
            if ( loss > 0.001 ) {
                Z backpropagate (error * 0.01)
            } else {
                print("Converged in $i iterations. Final loss=$loss, final Z=$Z")
                break
            }
        }
    }


    @Test
    fun xorLear() {
        val X = tensor(shape(2), 2, floatArrayOf(1f, 0f, 0f, 1f))
        val E = tensor(shape(2), 2, floatArrayOf(0f, 1f, 1f, 0f))

        val params = listOf<Tensor>(
            random(shape(2, 2)),
            random(shape(2, 2))
        ).map { it.makeMutable() }

        fun model(x: Tensor) = params[1] * params[0] * x
        learn(::model, X, E, 4)


    }
}