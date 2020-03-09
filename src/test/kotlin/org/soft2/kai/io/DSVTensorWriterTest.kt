package org.soft2.kai.io

import org.junit.Test
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.tensor
import java.io.StringWriter
import kotlin.test.assertEquals

class DSVTensorWriterTest {

    private val content = """
        0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
        0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
        0.0 1.5 3.0 4.5 6.0 7.5 9.0 10.5 12.0
    """.trimIndent()


    @Test
    fun writeContent() {
        val t1 = tensor(shape(3,3), 2) {
            if (it < 9) it / 2f else it - 9f
        }
        val t2 = tensor(shape(3,3)) { it.toFloat()*1.5f}
        val stringWriter = StringWriter()
        val writer = DSVTensorWriter(shape(3, 3), stringWriter)

        writer.write(t1)
        writer.write(t2)

        val s = stringWriter.buffer.toString().trimIndent()
        assertEquals(content, s)
    }
}