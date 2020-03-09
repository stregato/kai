package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.dot
import org.soft2.kai.tensors.tensor
import java.io.BufferedReader
import java.io.Reader

class DSVTensorReader(val shape: IntArray, reader: Reader, private val separator: String = "\\s+"): TensorReader {
    override var eof: Boolean = false

    private val reader = BufferedReader(reader)
    private val volume = shape.dot()

    override fun read(batchSize: Int): Tensor {
        val content = mutableListOf<Float>()

        for (i in 0 until batchSize) {
            val line = reader.readLine() ?: break
            val tokens = line.trim().split(Regex(separator))

            check( volume == tokens.size ) {
                "Mismatch between #items ${tokens.size} and volume $volume. Line is '$line'"
            }

            content.addAll(tokens.map { it.toFloat() })
        }

        val batch = content.size / volume
        eof = batch < batchSize
        return tensor(shape, batch, content.toFloatArray())
    }
}
