package org.soft2.kai.tensors.cpu

import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class MultiplicationByTensor {


    @Test
    fun multiplicationByOne() {

        val t = random(10, 10)

        assertEquals(t, t * eye(10))
        assertNotEquals(eye(10), t * eye(10))
    }


    @Test
    fun multiplicationByInv() {

        val t = tensor("1 2", "3 4")
        val ti = tensor("-2 1", "1.5 -0.5")

        assertEquals(eye(2), t * ti)
    }


    @Test
    fun multiplicationByTensor() {

        val a = tensor("11 3", "7 11")
        val b = tensor("8 0 1", "0 3 5")

        assertEquals(tensor("88 9 26", "56 33 62"), a*b)
        assertNotEquals(eye(3), a*b)
    }

    @Test
    fun multiplicationByOneWithBatch() {
        val a = random(4, 3, 3).shatter()
        assertEquals(a, a * eye(3))
    }

    @Test
    fun multiplicationWithBatch() {
        val a = tensor(4,3,3) { it.toFloat() }.shatter()
        val b = tensor("0 1", "2 3", "4 5")
        val e = tensor("10 13", "28 40", "46 67",
                              "64 94", "82 121", "100 148",
                              "118 175", "136 202", "1094 1404").reshape(2, 3)

        assertEquals(e, a * b)
    }



}