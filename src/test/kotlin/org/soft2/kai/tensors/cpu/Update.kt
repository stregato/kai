package org.soft2.kai.tensors.cpu

import org.junit.Before
import org.junit.Test
import org.soft2.kai.tensors.matrix
import org.soft2.kai.tensors.Kernel
import kotlin.test.assertEquals

open class Update {
    @Before
    open fun before() {
        Kernel.default = CpuKernel
    }


    @Test
    fun update() {
        val t = matrix("1 2 3", "4 5 6", "7 8 9").makeMutable()
        t.update(-0.5f, matrix("1 1 1", "1 1 1", "1 1 1"))

        assertEquals(matrix("0.5 1.5 2.5", "3.5 4.5 5.5", "6.5 7.5 8.5"), t )
    }

}