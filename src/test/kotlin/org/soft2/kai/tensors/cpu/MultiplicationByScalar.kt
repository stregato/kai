package org.soft2.kai.tensors.cpu

import org.junit.Test
import org.soft2.kai.tensor
import kotlin.test.assertEquals

class MultiplicationByScalar {


    @Test
    fun multiplicationByInt() {

        val t = tensor("1 2 3", "4 5 6", "7 8 9")
        val e = tensor("2 4 6", "8 10 12", "14 16 18")

        assertEquals(e, t * 2)
    }


    @Test
    fun multiplicationByFloat() {

        val t = tensor("1 2 3", "4 5 6", "7 8 9")
        val e = tensor("2 4 6", "8 10 12", "14 16 18")

        assertEquals(e, t * 2f)
    }


    @Test
    fun multiplicationWithBatch() {
        val t = tensor(10, 3, 3 ,3 ) { it.toFloat() / 2 }.reshape(3, 3, 3)
        val e = tensor(10, 3, 3 ,3 ) { it.toFloat() }.reshape(3, 3, 3)

        assertEquals(e, t * 2f)
    }

}