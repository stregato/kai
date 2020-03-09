package org.soft2.kai.tensors

import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/** create an array that represents a tensor shape */
fun shape(vararg dims: Int) = intArrayOf(*dims)

fun shapeToString(shape: IntArray) = "(${shape.joinToString() })"

fun elements(vararg es: Number) = es.map { it.toFloat() }.toFloatArray()

/** Get the mean of tensor elements */
fun mean(t: Tensor) = sum(t) / t.volume / t.batch

/** Get the sum of tensor elements */
fun sum(t: Tensor) = t.toFloatArray().sum()

/** Get the standard deviation of tensor elements */
fun std(t: Tensor): Float {
        val e = t.toFloatArray()
        val m = mean(t)
        val v = e.fold(0f) { acc, fl ->  acc + (fl-m).pow(2) }
        return sqrt(v/(e.size-1))
}

/** Get the norm L0 of tensor elements */
fun norm0(t: Tensor) = Tensor.kernel.norm0(t.handle)

/** Create a tensor given an array of floats */
fun tensor(shape: IntArray, batch: Int, elements: FloatArray): Tensor {
    check(shape.dot() * batch == elements.size) {
        "Size of element = ${elements.size} incompatible with volume ${shape.dot()} and batch size $batch"
    }
    return Tensor(shape, batch, elements)
}

fun tensor(shape: IntArray, elements: FloatArray) = tensor(shape, 1, elements)


/** Create a tensor with a generator*/
fun tensor(shape: IntArray, batch: Int=1,  gen: (i: Int) -> Float) =
    Tensor(shape, batch, (0 until shape.dot()*batch).map(gen).toFloatArray())

/** Create a tensor of one dimension and provided batch*/
fun vector(vararg elements: Float, batch: Int=1) = tensor(intArrayOf(elements.size), batch, elements)

/** Create a matrix */
fun matrix(vararg lines: String, batch: Int=1): Tensor {
    val rows = lines.map { line ->
        line.trim().split("\\s+".toRegex()).map { it.toFloat() }
    }

    val n = rows.size
    val m = rows.map { it.size }.max() ?: 1

    rows.forEachIndexed { idx, row ->
        assert(row.size == m) { "row $idx has less than $m items"}
    }

    return tensor(intArrayOf(n, m/batch), batch, rows.flatten().toFloatArray())
}




val eyeCache = mutableMapOf<Int, Tensor>()

/** Create a tensor identity with specified number of rows n */
fun eye(n: Int): Tensor {
    val shape = intArrayOf(n,n)
    fun make(): Tensor {
        val floatArray = FloatArray(shape.dot())
        for (i in 0 until n) {
            floatArray[i * n + i] = 1f
        }
        return tensor(shape, floatArray)
    }

    return if (n < 32) eyeCache.getOrPut(n) {make()} else make()
}


/** Create a tensor of zero elements with n rows and m columns */
fun zeros(shape: IntArray, batch: Int = 1) = Tensor(shape, batch, FloatArray(shape.dot()*batch))

/** Create a matrix of zero elements with n rows and m columns */
fun fill(alpha: Float, shape: IntArray, batch: Int = 1) = tensor(shape, batch) {alpha}

/** Create a matrix of random numbers between 0 and 1 */
fun random(shape: IntArray, batch: Int = 1): Tensor = tensor(shape, batch) { Random.nextFloat() }
