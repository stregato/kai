package org.soft2.kai.recipes

import org.soft2.kai.downloadFile
import org.soft2.kai.grad.learn
import org.soft2.kai.localFile
import org.soft2.kai.grad.pipe
import org.soft2.kai.io.*
import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.random
import org.soft2.kai.tensors.shape
import org.soft2.kai.tensors.tensor
import org.soft2.kai.tools.MNISTDataSource
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream


object mnist {

    private const val imagesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    private const val labelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    private const val hiddenLayerSize = 20
    private var params = listOf<Tensor>()

    private val model = pipe(
        { params[0] * it },
        { params[1] * it }
    )


    init {
        val paramsFile = localFile("mnist/params.bin")
        params = if ( paramsFile.exists() && paramsFile.length() > 0) {
            load(paramsFile)
        } else {
            train(paramsFile)
        }

    }

    fun match(width: Int, height: Int, image: FloatArray): Float {
        return model(
            tensor(shape(width*height), image)
        ).foFloat()
    }

    private fun load(paramsFile: File) = BinaryTensorReader(FileInputStream(paramsFile)).let {
        listOf(it.read(), it.read())
    }

    private fun train(paramsFile: File): List<Tensor> {
        val images = MNISTImagesReader(FileInputStream(downloadFile(imagesURL)))
        val labels = MNISTLabelsReader(FileInputStream(downloadFile(labelsURL)))

        val imageSize = images.n * images.m

        params = listOf(
            random(shape(hiddenLayerSize, imageSize)),
            random(shape(1, hiddenLayerSize))
        ).map { it.makeMutable() }

        learn(model, images, labels)

        BinaryTensorWriter(FileOutputStream(paramsFile)).let {
            it.write(params[0])
            it.write(params[1])
        }

        return params
    }

}

