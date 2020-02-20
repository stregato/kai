package org.soft2.kai

import org.soft2.kai.tensors.Cache
import org.soft2.kai.tensors.Tensor
import org.soft2.kai.tensors.prod
import kotlin.random.Random

/**
 * Create a matrix of zero elements with n rows and m columns
 */
fun zeros(vararg shape: Int) = Tensor(shape, FloatArray(shape.prod()))


/**
 * Create a matrix of zero elements with n rows and m columns
 */

fun tensor(vararg shape: Int, gen: (i: Int) -> Float) =
    Tensor(shape, (0 until shape.prod()).map(gen).toFloatArray())


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
        line.trim().split("\\s+".toRegex()).map { it.toFloat() }
    }

    val n = rows.size
    val m = rows.map { it.size }.max() ?: 0

    rows.forEachIndexed { idx, row ->
        assert(row.size == m) { "row $idx has less than $m items"}
    }

    val shape = if (n == 1) intArrayOf(m) else intArrayOf(n,m)

    return tensor(intArrayOf(n, m), rows.flatten().toFloatArray())
}

fun tensor(vararg elements: Float) =
    Tensor(intArrayOf(elements.size), elements)

fun tensor(vararg elements: Double) =
    Tensor(intArrayOf(elements.size), elements.map { it.toFloat() })

fun tensor(n: Int, m: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m), elements)

fun tensor(n: Int, m: Int, p: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m,p), elements)

fun tensor(n: Int, m: Int, p: Int, q: Int, vararg elements: Float) =
    Tensor(intArrayOf(n,m,p,q), elements)

fun tensor(shape: IntArray, elements: FloatArray) =
    Tensor(shape, elements)



val eyeCache = mutableMapOf<Int, Tensor>()

/**
 * Create a tensor identity with specified number of rows n
 */
fun eye(n: Int): Tensor {
    val shape = intArrayOf(n,n)
    fun make(): Tensor {
        val floatArray = FloatArray(shape.prod())
        for (i in 0 until n) {
            floatArray[i * n + i] = 1f
        }
        return Tensor(shape, floatArray)
    }

    return if (n < 32) eyeCache.getOrPut(n) {make()} else make()
}

/**
 * Create a matrix of random numbers between 0 and 1
 */
fun random(vararg shape: Int): Tensor =
    tensor(*shape) {
        Random.nextFloat()
    }
