package org.soft2.kai.io

import org.junit.Test
import org.soft2.kai.tensors.random
import org.soft2.kai.tensors.shape
import java.io.ByteArrayOutputStream

class TestBinaryReaderWriter {

    @Test
    fun writeRead() {
        val t = random(shape(3,3), 10)
        val output = ByteArrayOutputStream()
        val writer = BinaryTensorWriter(output)


    }
}