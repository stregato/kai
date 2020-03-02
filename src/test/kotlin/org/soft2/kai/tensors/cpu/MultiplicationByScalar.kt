package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensors.tensor
import org.soft2.kai.tensors.matrix
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.tensors.shape
import kotlin.test.assertEquals

open class MultiplicationByScalar {

    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }


    @Test
    fun multiplicationByInt() {

        val t = matrix("1 2 3", "4 5 6", "7 8 9")
        val e = matrix("2 4 6", "8 10 12", "14 16 18")

        assertEquals(e, t * 2)
    }


    @Test
    fun multiplicationByFloat() {

        val t = matrix("1 2 3", "4 5 6", "7 8 9")
        val e = matrix("2 4 6", "8 10 12", "14 16 18")

        assertEquals(e, t * 2f)
    }


    @Test
    fun multiplicationWithBatch() {
        val t = tensor(shape(10, 3, 3, 3)) { it.toFloat() / 2 }.shatter()
        val e = tensor(shape(10, 3, 3, 3)) { it.toFloat() }.shatter()

        assertEquals(e, t * 2f)
    }

}