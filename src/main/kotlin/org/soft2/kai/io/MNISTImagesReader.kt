package org.soft2.kai.io

import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.tensor
import org.soft2.kai.tools.MNISTDataSource
import java.io.DataInputStream
import java.io.InputStream
import java.lang.Integer.min
import java.util.zip.GZIPInputStream

class MNISTImagesReader(private val imagesStream: InputStream): TensorReader {

    private val dataInputStream = DataInputStream(GZIPInputStream(imagesStream))
    private val magicNumber = dataInputStream.readInt()
    private var numberOfItems = dataInputStream.readInt()/10
    val n = dataInputStream.readInt()
    val m = dataInputStream.readInt()
    private var pos = 0

    init {
        println("magic number is $magicNumber")
        println("number of items is $numberOfItems")
        println("number of rows is: $n")
        println("number of cols is: $m")
    }

    override fun read(batchSize: Int): Tensor {
        val batch = min(batchSize, numberOfItems)
        val elements = mutableListOf<Float>()

        for (i in 0 until batch) {
            for (j in 0 until n*m) {
                elements.add(dataInputStream.readUnsignedByte().toFloat())
            }
        }

        numberOfItems -= batch
        return tensor(shape(n,m), batch, elements.toFloatArray())
    }
}