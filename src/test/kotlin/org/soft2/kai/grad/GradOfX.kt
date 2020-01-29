package org.soft2.kai.grad

import org.junit.Before
import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.cpu.CpuKernel
import kotlin.test.assertEquals

class GradOfX {
    @Before
    fun before() {
        Kernel.default = CpuKernel
    }

    @Test
    fun gradOfWX() {
        val x = tensor("1 2 3", "4 5 6")
        val W = tensor("1 1", "2 2", "3 3")
        val y = x*W
        y.backpropagate(eye(2))

        assertEquals(W.t(), x.gradient)
        assertEquals(x.t(), W.gradient)

    }

    @Test
    fun gradOf3X() {

        val x = random(2,3)
        val y = x * 3f
        val z = tensor("1 0 0", "0 1 0")
        y.backpropagate(z)

        assertEquals(z*3f, x.gradient)

    }
}