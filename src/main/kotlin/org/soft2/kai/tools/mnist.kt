package org.soft2.kai.tools

import org.soft2.kai.downloadFile
import org.soft2.kai.tensor
import org.soft2.kai.tensors.Tensor
import java.io.DataInputStream
import java.io.FileInputStream
import java.io.InputStream
import java.util.zip.GZIPInputStream


//    fun print(index: Int): String {
//
//        val s = StringBuffer()
//
//        s.append("Label: $label")
//        s.appendln()
//        for ((i, f) in t.elements.withIndex()) {
//            s.append(
//                when(f.toInt()) {
//                    in 0..32 -> ' '
//                    in 33..64 -> '-'
//                    in 64..128 -> '+'
//                    else -> '#'
//                })
//            if ( (i + 1) % t.m == 0) {
//                s.appendln()
//            }
//        }
//        return s.toString()
//    }

class MNISTDataSource(val batchSize: Int,
                      val imagesStream: InputStream,
                      val labelsStream: InputStream): Iterator<MNISTDataSource.Data> {


    constructor(batchSize: Int,
                imagesFile: String,
                labelsFile: String):
            this(batchSize,
                FileInputStream(downloadFile(imagesFile)),
                FileInputStream(downloadFile(labelsFile)))

    class Data(val labels: FloatArray, val images: Tensor)


    private val dataInputStream = DataInputStream(GZIPInputStream(imagesStream))
    private val magicNumber = dataInputStream.readInt()
    private val numberOfItems = dataInputStream.readInt()/10
    val n = dataInputStream.readInt()
    val m = dataInputStream.readInt()

    private val labelInputStream = DataInputStream(GZIPInputStream(labelsStream))
    private val labelMagicNumber = labelInputStream.readInt()
    private val numberOfLabels = labelInputStream.readInt()
    private var currentPos = 0

    init {
        println("magic number is $magicNumber")
        println("number of items is $numberOfItems")
        println("number of rows is: $n")
        println("number of cols is: $m")

        println("labels magic number is: $labelMagicNumber")
        println("number of labels is: $numberOfLabels")

    }

    override fun next(): Data {
        val dataSize = if ( currentPos + batchSize > numberOfItems ) numberOfItems-currentPos else batchSize
        val labels = FloatArray(dataSize)
        val images =  Array(dataSize) { FloatArray(n*m) }


        for (i in currentPos until currentPos+dataSize) {
            for (j in 0 until n*m) {
                images[i][j] = dataInputStream.readUnsignedByte().toFloat()
            }

            labels[i] = labelInputStream.readUnsignedByte().toFloat()
        }
        currentPos += dataSize
        return Data(labels, tensor(intArrayOf(n,m), images.flatMap { it.toList() }.toFloatArray()))
    }

    override fun hasNext() = currentPos < numberOfItems

}

object MNIST {
    fun training(batchSize: Int = 32) =
        MNISTDataSource(batchSize,
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")

    fun test(batchSize: Int = 32) =
        MNISTDataSource(batchSize,
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")

}




