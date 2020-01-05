package org.soft2.kai.tensors.cpu

import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class BitwiseMultiplication {


    @Test
    fun multiplicationBitwise() {
        val a = tensor("1 2", "3 4")
        val e = tensor("1 4", "9 16")

        assertEquals(e, a bitwise a)

    }


    @Test
    fun multiplicationBitwiseBatch() {
        val a = tensor(3, 2, 2 ) { it.toFloat() / 2 }.reshape(2, 2)
        val b = tensor(2, 2 ,2 ) { it.toFloat() }.reshape(2, 2)
        val e = tensor(3, 2, 2 ) { it.toFloat() / 2 * (it % 8) }.reshape(2, 2)

        assertEquals(e, a bitwise b)

    }
}