package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Kernel
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class BitwiseMultiplication {

    @Before
    fun before() {
        Kernel.default = CpuKernel
    }


    @Test
    fun multiplicationBitwise() {
        val a = tensor("1 2", "3 4")
        val e = tensor("1 4", "9 16")

        assertEquals(e, a bitwise a)

    }


    @Test
    fun multiplicationBitwiseBatch() {
        val a = tensor(3, 2, 2 ) { it.toFloat() / 2 }.shatter()
        val b = tensor(2, 2 ,2 ) { it.toFloat() }.shatter()
        val e = tensor(3, 2, 2 ) { it.toFloat() / 2 * (it % 8) }.shatter()

        assertEquals(e, a bitwise b)

    }
}