package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.dot
import org.soft2.kai.tensors.shapeToString
import org.soft2.kai.tensors.tensor
import java.io.DataInputStream
import java.io.EOFException
import java.io.InputStream
import java.util.zip.ZipInputStream

class BinaryTensorReader(stream: InputStream, private val compress: Boolean = true): TensorReader {
    private val reader = if (compress) DataInputStream(ZipInputStream(stream)) else DataInputStream(stream)

    override var eof: Boolean = false

    private fun readShape(): IntArray {
        val shapeSize = reader.readInt()
        return IntArray(shapeSize) { reader.readInt() }
    }

    override fun read(batchSize: Int): Tensor {
        if ( reader.available() == 0 ) {
            return tensor(intArrayOf(0)) {0f}
        }

        val content = mutableListOf<Float>()
        val shape = readShape()
        val volume = shape.dot()

        for (i in 0 until batchSize) {
            for (j in 0 until volume ) {
                content.add(reader.readFloat())
            }

            try {
                val s = readShape()
                check( s.contentEquals(shape) ) {
                    """Unexpected shape ${shapeToString(s)} in read of batch $batchSize. 
                       Expected shape is ${shapeToString(shape)}""".trimIndent()
                }
            } catch (e: EOFException) {
                eof = true
                break
            }
        }

        val batch = content.size / volume
        return tensor(shape, batch, content.toFloatArray())
    }

}
