package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.dot
import java.io.*
import java.util.zip.ZipOutputStream

class BinaryTensorWriter(stream: OutputStream, private val compress: Boolean = true): TensorWriter {
    private val writer = if (compress) DataOutputStream(ZipOutputStream(stream)) else DataOutputStream(stream)

    private fun writeHeader(t: Tensor) {
        writer.writeInt(t.shape.size)
        t.shape.forEach { writer.writeInt(it) }
    }

    override fun write(t: Tensor) {
        val volume = t.shape.dot()
        t.toFloatArray().forEachIndexed {i, f ->
            if ( i % volume == 0 ) {
                writeHeader(t)
            }
            writer.writeFloat(f)
        }
        writer.flush()
    }
}
