package org.soft2.kai.recipes

import org.junit.Test

class mnisttest {

    @Test
    fun simple() {
        mnist.match(10,10, FloatArray(100) {0f})
    }
}