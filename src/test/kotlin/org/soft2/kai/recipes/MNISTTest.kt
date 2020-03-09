package org.soft2.kai.recipes

import org.junit.Test

class MNISTTest {

    @Test
    fun simple() {
        mnist.match(10,10, FloatArray(100) {0f})
    }
}