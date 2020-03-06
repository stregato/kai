package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.dot
import java.io.BufferedWriter
import java.io.Writer

class TextTensorWriter(val shape: IntArray, writer: Writer, private val separator: String = " "): TensorWriter {
    private val writer = BufferedWriter(writer)
    private val volume = shape.dot()

    override fun write(t: Tensor) {
        check( volume == t.volume) {
            "Tensor has volume ${t.volume} and volume $volume was expected. Tensor content is $t"
        }

        for ((i, f) in t.toFloatArray().withIndex()) {
            if ( (i + 1) % volume == 0 ) {
                writer.appendln(f.toString())
            } else {
                writer.write("$f$separator")
            }
        }
        writer.flush()
    }
}
