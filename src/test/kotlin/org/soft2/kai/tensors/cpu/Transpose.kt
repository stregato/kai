package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Kernel
import kotlin.test.assertEquals

open class Transpose {
    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }

    @Test
    fun transpose2() {
        val t = tensor("1 2", "3 4")
        val e = tensor("1 3", "2 4")
        assertEquals(e, t.t())
    }

    @Test
    fun transpose3() {
        val t = tensor("0 3 6", "1 4 7", "2 5 8")
        val e = tensor("0 1 2", "3 4 5", "6 7 8")
        assertEquals(e, t.t())
    }

    @Test
    fun transpose4() {
        val t = tensor("0 4 8 12", "1 5 9 13", "2 6 10 14", "3 7 11 15")
        val e = tensor("0 1 2 3", "4 5 6 7", "8 9 10 11", "12 13 14 15")
        assertEquals(e, t.t())
    }


}