package org.soft.kai

import org.soft.kai.tensors.Tensor
import org.soft.kai.tensors.dot
import kotlin.random.Random

/**
 * Create a matrix of zero elements with n rows and m columns
 */
fun zeros(vararg shape: Int) = Tensor(shape, FloatArray(shape.dot()))


/**
 * Create a matrix of zero elements with n rows and m columns
 */

fun tensor(vararg shape: Int, immutable: Boolean = false, gen: (i: Int) -> Float) =
    Tensor(shape, (0 until shape.dot()).map(gen).toFloatArray())


fun tensor(range: ClosedFloatingPointRange<Float>, step: Float = 1f): Tensor {
    val start = range.start
    val end = range.endInclusive
    val size = ((end - start) / step).toInt() + 1

    return Tensor(
        intArrayOf(size),
        (0 until size).map { start + step * it }.toFloatArray()
    )
}

fun tensor(vararg lines: String): Tensor {
    val rows = lines.map { line ->
        line.split("\\s+".toRegex()).map { it.toFloat() }
    }

    val n = rows.size
    val m = rows.map { it.size }.max() ?: 0

    rows.forEachIndexed { idx, row ->
        assert(row.size == m) { "row $idx has less than $m items"}
    }

    return tensor(intArrayOf(n, m), rows.flatten().toFloatArray())
}

fun tensor(vararg elements: Float) =
    Tensor(intArrayOf(elements.size), elements)

fun tensor(n: Int, m: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m), elements)

fun tensor(n: Int, m: Int, p: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m,p), elements)

fun tensor(n: Int, m: Int, p: Int, q: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m,p,q), elements)

fun tensor(shape: IntArray, elements: FloatArray) =
    Tensor(shape, elements)



/**
 * Create a tensor identity with specified number of rows n
 */
fun eye(n: Int): Tensor {
    val shape = intArrayOf(n,n)

    val floatArray = FloatArray(shape.dot())
    for (i in 0 until n) {
        floatArray[i * n + i] = 1f
    }

    return Tensor(shape, floatArray)
}

/**
 * Create a matrix of random numbers between 0 and 1
 */
fun random(vararg shape: Int): Tensor =
    tensor(*shape) {
        Random.nextFloat()
    }
