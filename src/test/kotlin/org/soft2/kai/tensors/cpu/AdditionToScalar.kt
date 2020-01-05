package org.soft2.kai.tensors.cpu

import org.junit.Test
import org.soft2.kai.tensor
import kotlin.test.assertEquals

class AdditionToScalar {


    @Test
    fun additionToInt() {

        val t = tensor("1 2 3", "4 5 6", "7 8 9")
        val e = tensor("3 4 5", "6 7 8", "9 10 11")

        assertEquals(e, t + 2)
    }

    @Test
    fun additionToFloat() {

        val t = tensor("1 2 3", "4 5 6", "7 8 9")
        val e = tensor("4.15 5.15 6.15", "7.15 8.15 9.15", "10.15 11.15 12.15")

        assertEquals(e, t + 3.15f)
    }

    @Test
    fun additionOnBatch() {
        val t = tensor(10, 3, 3 ,3 ) { it.toFloat() / 2 }.reshape(3, 3, 3)
        val e = tensor(10, 3, 3 ,3 ) { it.toFloat() / 2 + 3.15f}.reshape(3, 3, 3)

        assertEquals(e, t + 3.15f)
    }



}