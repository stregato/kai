package org.soft.kai.tensors

import org.soft.kai.grad.*
import kotlin.math.pow
import kotlin.math.sqrt


/**
 * Matrix that supports basic operations
 */


open class Tensor(val shape: IntArray, internal val handle: Handle) {

    var gradient: Gradient? = null

    companion object {
        var kernel: Kernel = Kernel.get()
    }

    constructor(shape: IntArray, elements: FloatArray): this(shape, kernel.allocate(elements))

    private val volume = shape.dot()

    /** when greater than 1, the tensor is actually a batch of defined shape */
    val batch = kernel.size(handle) / volume

    val mean: Float
        get() = sum / volume / batch

    val sum: Float
        get() = toFloatArray().sum()

    val std: Float
        get() {
            val e = toFloatArray()
            val m = e.sum() / volume / batch
            val v = e.fold(0f) { acc, fl ->  acc + (fl-m).pow(2) }
            return sqrt(v/(e.size-1))
        }


    fun hasElements(vararg e: Float) = kernel.toFloatArray(handle).contentEquals(e)

    fun hasElements(s: String) = hasElements(
        *s.split("\\s+".toRegex()).map { it.toFloat() }.toFloatArray())

    fun toFloatArray() = kernel.toFloatArray(handle)

    /**
     * Convert a tensor to float. It requires to tensor to have volume 1 and batch 1
     */
    fun foFloat(): Float {
        check(volume == 1 && batch == 1) {
            "Tensor $this has size or batch different than 1"
        }
        return toFloatArray()[0]
    }

    /**
     * Create a new tensor with same elements but different shape
     */
    fun reshape(vararg s: Int) = Tensor(s, handle)

    infix fun dot (t: Tensor) = grad(this, t, TensorMulGrad()) { _, t ->
        require(volume == t.volume) {
            "Matrix $this and $t have not the same volume"
        }

        val h = kernel.dot(handle, t.handle)
        Tensor(intArrayOf(1), h)
    }

    operator fun plus(t: Tensor): Tensor {
        require(shape.contentEquals(t.shape)) {
            "Shape mismatch in adding two tensors $this and $t"
        }

        val h = kernel.add(handle, t.handle)
        return Tensor(shape, h)
    }

    operator fun minus(t: Tensor): Tensor {
        require(shape.contentEquals(t.shape)) {
            "Shape mismatch in adding two tensors $this and $t"
        }

        val h = kernel.add(handle, t.handle, -1f)
        return Tensor(shape, h)
    }

    operator fun times(t: Tensor): Tensor {
        check(shape.size < 3) {
            "Invalid shape in tensor $this for matrix multiplication"
        }
        require(t.shape.size < 3) {
            "Invalid shape in tensor $t for matrix multiplication"
        }
        require ( shape[1] == t.shape[0] ) {
            "Incompatible shape size between $this and $t"
        }
        require( batch == t.batch || batch == 1 || t.batch == 1) {
            "Incompatible batch between $this and $t"
        }
        val n = shape[0]
        val m = if (shape.size > 1) shape[1] else 1
        val q = if (t.shape.size > 1) t.shape[1] else 1

        val s = intArrayOf(shape[0], t.shape[1])
        val c = kernel.matrixMul(handle, t.handle, n, m, q)
        return Tensor(s, c)
    }

    fun mutable() = MutableTensor(shape, handle)


    operator fun times(s: Float) = grad(this, ScalarMulGrad(s)) {
            x-> x.map {it *s }
    }

    operator fun div(s: Float) = times(1/s)

    override fun toString() = StringBuilder()
        .append("Tensor $kernel (").append(shape.joinToString("•")).append("): [")
        .append(toFloatArray().joinToString()).append("]").toString()

    operator fun get(vararg x: Int): Float {
        require(x.size == shape.size || x.size == shape.size+1)
        return kernel.get(handle, loc(*x))
    }

    fun map(transform: (Float) -> Float) = Tensor(shape,
        toFloatArray().map(transform).toFloatArray()
    )

    fun t(): Tensor {
        assert(shape.size <= 2) {
            "Transpose is supported only for vector and matrix. $this has shape size ${shape.size}"
        }

        val n = shape[0]
        val m = shape.getOrElse(1) { 1 }
        val ns = shape.reversed().toIntArray()
        val nh = if ( m == 1 ) handle else kernel.transpose(handle, n, m)

        return Tensor(ns, nh)
    }


    private fun loc(vararg x: Int): Int {
        val batchIndex = x.size == shape.size+1
        var loc = if (batchIndex) x[0]*volume else 0
        var factor = volume
        val indices = if (batchIndex) x.takeLast(x.size-1).indices else x.indices
        for (idx in indices) {
            factor /= shape[idx]
            loc += factor * x[idx]
        }
        return loc
    }

    protected fun finalize() {
        kernel.release(handle)
    }
}

operator fun Float.times(t: Tensor): Tensor = t * this

/**
 * Returns the sum of all elements in the array.
 */
@kotlin.jvm.JvmName("sumOfInt")
fun IntArray.dot(): Int {
    var dot = 1
    for (element in this) {
        dot *= element
    }
    return dot
}

