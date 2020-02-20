package org.soft2.kai.grad

import org.junit.Before
import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cpu.CpuKernel

class BackPropagation {
    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }

    @Test
    fun simpleBack() {
        val X = tensor("1 1 1")
        val W = random(3, 3).makeMutable()
        val E = tensor("-1 0 1")
        var i = 0

        while (true) {
            i++
            val Y = X * W
            val gap = ( Y - E ) * 0.1
            if ( gap.norm0 > 0.001 ) {
                Y backpropagate gap
            } else {
                print("Converged in $i iterations. Final gap=$gap, Y=$Y, W=$W")
                break
            }
        }
    }

    @Test
    fun doubleBack() {
        val X = tensor("1 1 1")
        val W1 = random(3, 3).makeMutable()
        val W2 = random(3, 3).makeMutable()
        val E = tensor("-1 0 1")
        var i = 0

        while (true) {
            i++
            val Y = X * W1
            val Z = Y * W2
            val gap = ( Z - E ) * 0.1
            if ( gap.norm0 > 0.001 ) {
                Z backpropagate gap
            } else {
                print("Converged in $i iterations. Final gap=$gap, final Z=$Z")
                break
            }
        }
    }

    @Test
    fun xor() {
        val X = tensor("1 0", "0 1") batchOf 2
        val W1 = random(2, 2).makeMutable()
        val W2 = random(2, 2).makeMutable()
        val E = tensor("0 1", "1 0") batchOf 2
        var i = 0

        while (true) {
            i++
            val Y = W1 * X
            val Z = W2 * Y
            val error = Z - E
            val loss = error.norm0
            if ( loss > 0.001 ) {
                Z backpropagate (error * 0.01)
            } else {
                print("Converged in $i iterations. Final loss=$loss, final Z=$Z")
                break
            }
        }
    }

}