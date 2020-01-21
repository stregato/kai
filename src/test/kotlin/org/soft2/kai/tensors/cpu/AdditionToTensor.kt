package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.eye
import org.soft2.kai.random
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Kernel
import org.soft2.kai.zeros
import kotlin.test.assertEquals

open class AdditionToTensor {

    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }


    @Test
    fun additionToZero() {
        val a = random(10, 7)
        val b = zeros(*a.shape)

        assertEquals(a, a + b)
    }

    @Test
    fun additionToOne() {
        val a = tensor("1 2 3", "4 5 6", "7 8 9")
        val e = tensor("2 2 3", "4 6 6", "7 8 10")

        assertEquals(e, a + eye(3))
    }

    @Test
    fun additionToSelf() {
        val a = random(10, 7)

        assertEquals(a*2, a+a)
    }

    @Test
    fun additionWithBatch() {
        val a = tensor(3, 2, 2 ) { it.toFloat() / 2 }.shatter()
        val b = tensor(2 ,2 ) { it.toFloat() }
        val e = tensor(3, 2, 2 ) { it.toFloat() / 2 + it % 4 }.shatter()

        assertEquals(e, a+b)
    }

    @Test
    fun additionWithDifferentBatch() {
        val a = tensor(4, 2, 2 ) { it.toFloat() / 2 }.shatter()
        val b = tensor(2, 2 ,2 ) { it.toFloat() }.shatter()
        val e = tensor(4, 2, 2 ) { it.toFloat() / 2 + it % 8 }.shatter()

        assertEquals(e, a+b)
    }


}