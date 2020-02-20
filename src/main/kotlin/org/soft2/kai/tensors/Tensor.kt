package org.soft2.kai.tensors

import org.soft2.kai.eyeCache
import org.soft2.kai.grad.*
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt


/**
 * Matrix that supports basic operations
 */


open class Tensor(val shape: IntArray, internal val handle: Handle) {

    var flow = arrayOf<Step>()
    var mutable = false

    companion object {
        var kernel: Kernel = Kernel.default
    }

    constructor(shape: IntArray, elements: FloatArray): this(shape, kernel.allocate(elements))

    private val volume = shape.prod()

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

    val norm0: Float
        get() {
            return kernel.norm0(handle)
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

    infix fun batchOf(batchSize: Int): Tensor {
        val d = shape.last()/batchSize
        val s = if ( d == 1 ) shape.dropLast(1).toIntArray() else shape.mapIndexed { index, i ->
            if (index < shape.size-1 ) i else d
        }.toIntArray()
        return Tensor(s, handle)
    }

    fun makeMutable(): Tensor {
        check( !eyeCache.containsValue(this) ) {
            "Cached tensors cannot be mutable. Do not use eye for mutable tensors"
        }
        mutable = true
        return this
    }

    fun expand(vararg targetShape: Int): Tensor {
        val targetVolume = targetShape.prod()
        check( targetVolume > volume && targetVolume % volume == 0) {
            "target shape must be multiple of shape"
        }

        return Tensor(targetShape, kernel.expand(handle, volume, targetVolume))
    }

    /**
     * Create a new tensor with same elements but different shape
     */
    fun reshape(vararg s: Int) = Tensor(s, handle)

    /**
     * Collapse a tensor dimensions to batch
     */
    fun shatter(dims: Int = 1) = Tensor(shape.drop(dims).toIntArray(), handle)

    /**
     * Unsqueeze a tensor converting the batch to a dimension in the shape
     */
    fun unshatter() = Tensor(intArrayOf(batch, *shape), handle)


    operator fun rem(t: Tensor) {
        require(volume == t.volume) {
            "Matrix $this and $t have not the same volume"
        }

        val h = kernel.bwMul(handle, t.handle)
        Tensor(intArrayOf(1), h)
    }

    operator fun plus(f: Float) =
        Tensor(shape,  kernel.toFloatArray(handle).map { it + f }.toFloatArray())
            .trace(this, "Scalar Addition") { it }

    operator fun plus(d: Double) = plus(d.toFloat())

    operator fun plus(i: Int) = plus(i.toFloat())

    operator fun plus(t: Tensor): Tensor {
        require(shape.contentEquals(t.shape)) {
            "Shape mismatch in adding two tensors $this and $t"
        }

        val h = kernel.add(handle, t.handle, 1f)
        return Tensor(shape, h).trace(this, "Addition 0", {it}, t, "Addition 1", {it})
    }

    operator fun minus(f: Float) =
        Tensor(shape,  kernel.toFloatArray(handle).map { it - f }.toFloatArray())
            .trace(this, "Float Subtraction") { it * -1f }

    operator fun minus(d: Double) = minus(d.toFloat())

    operator fun minus(i: Int) = minus(i.toFloat())

    operator fun minus(t: Tensor): Tensor {
        require(shape.contentEquals(t.shape)) {
            "Shape mismatch in adding two tensors $this and $t"
        }

        val h = kernel.add(handle, t.handle, -1f)
        return Tensor(shape, h)
    }


    operator fun times(i: Int) = times(i.toFloat())

    operator fun times(t: Tensor): Tensor {
        check(shape.size < 3) {
            "Invalid shape in tensor $this for matrix multiplication"
        }
        require(t.shape.size < 3) {
            "Invalid shape in tensor $t for matrix multiplication"
        }
        require ( shape.getOrElse(1) {1} == t.shape[0] ) {
            "Incompatible shape size between $this and $t"
        }
        require( batch == t.batch || batch == 1 || t.batch == 1) {
            "Incompatible batch between $this and $t"
        }

        val n = shape[0]
        val m = if (shape.size > 1) shape[1] else 1
        val q = if (t.shape.size > 1) t.shape[1] else 1

        val s = if (q > 1 ) intArrayOf(n, q) else intArrayOf(n)
        val c = kernel.mul(handle, t.handle, n, m, q)
        return Tensor(s, c).trace(this, "Multiplication 0", { it * t.t()},
                                   t, "Multiplication 1", { this.t() * it } )
    }

    operator fun times(alpha: Double) = times(alpha.toFloat())

    operator fun times(alpha: Float): Tensor {
        return Tensor(shape, kernel.mul(handle, alpha))
            .trace(this, "Scalar Multiplication") {
                Tensor(shape, kernel.mul(it.handle, alpha))
            }
    }

    operator fun div(s: Float) = times(1/s)

    infix fun bitwise(t: Tensor): Tensor {
        require(shape.contentEquals(t.shape)) {
            "Bitwise requires same shape"
        }

        val tfs = kernel.toFloatArray(t.handle)
        val fs = kernel.toFloatArray(handle)
        val size = max(tfs.size, fs.size)

        return Tensor(shape, (0 until size).map {
            fs[it % fs.size]*tfs[it % tfs.size]
        }.toFloatArray())
    }

    override fun toString() = StringBuilder()
        .append("Tensor $kernel (").append(batch).append("x").append(shape.joinToString("•")).append("): [")
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
        val ns = intArrayOf(m, n)
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

    override fun equals(other: Any?) =
        if ( other is Tensor )
            shape.contentEquals(other.shape) && kernel.toFloatArray(handle).contentEquals(kernel.toFloatArray(other.handle))
        else
            false


    fun update(alpha: Float, A: Tensor) {
        check( mutable ) {
            "Update is invalid on immutable tensors. Call makeMutable to make a tensor mutable"
        }
        kernel.update(handle, alpha, A.handle)
    }


    protected fun finalize() {
        kernel.release(handle)
    }

    override fun hashCode(): Int {
        var result = shape.contentHashCode()
        result = 31 * result + handle.hashCode()
        return result
    }


}

operator fun Float.times(t: Tensor): Tensor = t * this

/**
 * Returns the sum of all elements in the array.
 */
@kotlin.jvm.JvmName("sumOfInt")
fun IntArray.prod(): Int {
    var prod = 1
    for (element in this) {
        prod *= element
    }
    return prod
}
