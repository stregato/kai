package org.soft2.kai.input

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.dot
import org.soft2.kai.tensors.tensor
import java.io.BufferedReader
import java.io.Reader

class TextTensorInput(val shape: IntArray, reader: Reader, private val separator: String = ' '): TensorInput {
    private val reader = BufferedReader(reader)
    private val volume = shape.dot()

    override fun read(batchSize: Int): Tensor {
        val content = MutableList<Float>(0)

        for (i in 0 until batchSize) {
            val line = reader.readLine() ?: break
            content.addAll(line.split(separator).map { it.toFloat() })
        }

        val batch = content.size / volume
        return tensor(shape, batch, content.toFloatArray())
    }
}
