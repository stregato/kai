package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.tensor
import org.soft2.kai.tools.MNISTDataSource
import java.io.DataInputStream
import java.io.InputStream
import java.lang.Integer.min

import java.util.zip.GZIPInputStream

class MNISTLabelsReader(private val labelsStream: InputStream): TensorReader {

    private val labelInputStream = DataInputStream(GZIPInputStream(labelsStream))
    private val labelMagicNumber = labelInputStream.readInt()
    private var numberOfLabels = labelInputStream.readInt()

    init {
        println("labels magic number is: $labelMagicNumber")
        println("number of labels is: $numberOfLabels")
    }

    override fun read(batchSize: Int): Tensor {
        val batch = min(batchSize, numberOfLabels)
        val labels = mutableListOf<Float>()

        for (i in 0 until batch) {
            labels.add(labelInputStream.readUnsignedByte().toFloat())
        }

        numberOfLabels -= batch
        return tensor(shape(1), batch, labels.toFloatArray())
    }
}