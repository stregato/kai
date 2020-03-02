package org.soft2.kai.grad

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensors.*
import org.soft2.kai.tensors.cpu.CpuKernel
import kotlin.test.assertEquals

open class CPUGradient {
    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }

    @Test
    fun gradOfWX() {
        val x = matrix("1 2 3", "4 5 6")
        val W = matrix("1 1", "2 2", "3 3")
        val y = x*W
        val grad = y gradient eye(2)

        assertEquals(W.t(), grad[0])
        assertEquals(x.t(), grad[1])

    }

    @Test
    fun gradOfAdd() {
        val x = matrix("1 2 3", "4 5 6")
        val y = matrix("1 1 1", "2 2 2")
        val z = x + y
        val e = tensor(intArrayOf(2, 3)) { 1f }
        val grad = z gradient e

        assertEquals(e, grad[0])
        assertEquals(e, grad[1])
    }


    @Test
    fun gradOf3X() {
        val x = random(shape(2, 3))
        val y = x * 3f
        val z = matrix("1 0 0", "0 1 0")
        val grad = y gradient  z

        assertEquals(z*3f, grad[0])
    }
}