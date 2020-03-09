package org.soft2.kai.io

import org.junit.Test
import org.soft2.kai.tensors.shape
import java.io.StringReader
import kotlin.test.assertEquals

class DSVTensorReaderTest {

    private val content = """
        0 0.5 1 1.5 2 2.5 3 3.5 4
        0 1 2 3 4 5 6 7 8
        0 1.5 3 4.5 6 7.5 9 10.5 12 
    """.trimIndent()


    @Test
    fun loadContent() {
        val reader = DSVTensorReader(shape(3, 3), StringReader(content))

        val t1 = reader.read(2)
        assertEquals(2, t1.batch)

        val t2 = reader.read(2)
        assertEquals(1, t2.batch)
        
        val t3 = reader.read(2)
        assertEquals(0, t3.batch)
    }
}