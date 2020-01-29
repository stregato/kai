package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.random
import org.soft2.kai.tensor
import kotlin.test.assertEquals

class GradOfX {



    @Test
    fun gradOfX() {

        val x = random(2,3)
        x * 1f


        val z = tensor("1 0 0", "0 1 0")

        assertEquals(z, x.gradient(z))

    }
}