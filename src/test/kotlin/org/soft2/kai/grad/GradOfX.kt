package org.soft2.kai.grad

import org.junit.Test
import org.soft2.kai.random
import org.soft2.kai.tensor
import kotlin.test.assertEquals

class GradOfX {



    @Test
    fun gradOfX() {

        val x = random(2,3)
        val y = x * 3f
        val z = tensor("1 0 0", "0 1 0")
        y.backpropagate(z)

        assertEquals(z*3f, x.gradient)

    }
}