package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensors.tensor
import org.soft2.kai.tensors.matrix
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.shape
import kotlin.test.assertEquals

class BitwiseMultiplication {

    @Before
    fun before() {
        Kernel.default = CpuKernel
    }


    @Test
    fun multiplicationBitwise() {
        val a = matrix("1 2", "3 4")
        val e = matrix("1 4", "9 16")

        assertEquals(e, a bitwise a)

    }


    @Test
    fun multiplicationBitwiseBatch() {
        val a = tensor(shape(3, 2, 2)) { it.toFloat() / 2 }.shatter()
        val b = tensor(shape(2, 2, 2)) { it.toFloat() }.shatter()
        val e = tensor(shape(3, 2, 2)) { it.toFloat() / 2 * (it % 8) }.shatter()

        assertEquals(e, a bitwise b)

    }
}