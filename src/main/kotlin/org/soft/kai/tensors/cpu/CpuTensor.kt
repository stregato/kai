//package org.soft.kai.tensors.cpu
//
//
//import org.soft.kai.tensors.Tensor
//import org.soft.kai.tensors.dot
//import org.soft.kai.toString
//import kotlin.math.sqrt
//
///**
// * Matrix that supports basic operations
// */
//
//fun useCpu(): Boolean {
//    Tensor.create = { s, e -> CpuTensor(s, e) }
//    return false
//}
//
//
//open class CpuTensor(
//    final override val shape: IntArray,
//    open var elements: FloatArray
//): Tensor {
//
//    val volume = shape.dot()
//
//    override val batch = elements.size / volume
//
//    override val mean: Float
//        get() = sum / volume
//
//    override val sum: Float
//        get() {
//            return elements.sum()
//        }
//
//    override val std: Float
//        get() {
//            val mn = mean
//            val lst = toFloatArray().map { it - mn }
//            val sqSize = sqrt(volume.toDouble())
//
//            return Tensor.create(shape, lst.toFloatArray()).norm2 / sqSize.toFloat()
//        }
//
//    override val scalar: Float
//        get() {
//            check(shape.size == 1) {
//                "Tensor $this is not a scalar"
//            }
//            return toFloatArray()[0]
//        }
//
//    override val norm2: Float
//        get() = 0f
//
//    override fun toFloatArray() = elements
//
//    override fun hasElements(vararg e: Float) = toFloatArray().contentEquals(e)
//
//    override fun hasElements(s: String) =
//        elements.contentEquals(
//            s.split("\\s+".toRegex()).map { it.toFloat() }.toFloatArray())
//
//
//
//    override fun reshape(vararg s: Int) = Tensor.create(s, elements)
//
//    override operator fun unaryMinus() =
//        Tensor.create(shape, elements.map { it * -1f }.toFloatArray())
//
//    override operator fun plus(t: Tensor): Tensor = add(1f, t)
//
//    override operator fun minus(t: Tensor): Tensor = add(-1f, t)
//
//    override operator fun plus(f: Float) =
//        Tensor.create(shape, elements.map {it + f}.toFloatArray())
//
//    override operator fun minus(f: Float) =
//        Tensor.create(shape, elements.map {it - f}.toFloatArray())
//
//    override operator fun plusAssign(t: Tensor) {
//        elements.indices.forEach { i ->
//            elements[i] += (t as CpuTensor).elements[i]
//        }
//    }
//
//    override operator fun times(t: Tensor): Tensor {
//        val b = t as CpuTensor
//
//        check ( shape.size < 3 ) {
//            "Unsupported shape ${toString(shape)} on first argument"
//        }
//        check ( b.shape.size < 3 ) {
//            "Unsupported shape ${toString(t.shape)} on second argument"
//        }
//        check ( shape[0] == b.shape[0] ) {
//            "Incompatible shape size between ${toString(shape)} and ${toString(b.shape)}"
//        }
//        check( batch == b.batch || batch == 1 || b.batch == 1) {
//            "Incompatible batch $batch and ${b.batch}. Batches must be the same of one must be 1"
//        }
//
//
//        val n = shape[0]
//        val m = if (shape.size > 1) shape[1] else 1
//
//        val q = if (t.shape.size > 1) b.shape[1] else 1
//        val output = FloatArray(m*q)
//        val stripeA = if (batch > 1) volume else 0
//        val stripeB = if (b.batch > 1) b.volume else 0
//
//
//        for (h in 0 until maxOf(batch, b.batch)) {
//            for (i in 0 until n) {
//                for (j in 0 until q) {
//                    var s = 0f
//                    for (k in 0 until m) {
//                        s += elements[h * stripeA + i * n + k] *
//                                b.elements[h * stripeB + k * q + j]
//                    }
//                    output[h * n * q + i * q + j] = s
//                }
//            }
//        }
//
//        return Tensor.create(intArrayOf(n,q), output)
//    }
//
//
//    override operator fun times(s: Float): Tensor {
//        return Tensor.create(shape, elements.map { it*s }.toFloatArray() )
//    }
//
//
//    override operator fun div(s: Float): Tensor {
//        return times(1 / s)
//    }
//
//    override fun toString(): String {
//        val sb = StringBuilder()
//        val m = if (shape.size > 1) shape[1] else Int.MAX_VALUE
//
//        elements.forEachIndexed { index, fl ->
//            if (index % m == 0) {
//                sb.appendln()
//            }
//            sb.append(fl).append(' ')
//        }
//        return sb.toString()
//    }
//
//    override operator fun get(vararg x: Int): Float {
//        check( shape.size == x.size || shape.size+1 == x.size) {
//            "Invalid selector of size ${x.size}. It should be ${shape.size} or ${shape.size+1} for batch tensors"
//        }
//        return elements[loc(*x)]
//    }
//
//
//    private fun loc(vararg x: Int): Int {
//        val batchIndex = x.size == shape.size+1
//        var loc = if (batchIndex) x[0]*volume else 0
//        var factor = volume
//        val indices = if (batchIndex) x.takeLast(x.size-1).indices else x.indices
//        for (idx in indices) {
//            factor /= shape[idx]
//            loc += factor * x[idx]
//        }
//        return loc
//    }
//
//
//    protected open fun add(alpha: Float, t: Tensor): Tensor {
//        if (!shape.contentEquals(t.shape)) {
//            throw Exception("Incompatible t sizes ${toString(t.shape)} != ${toString(shape)}")
//        }
//        return Tensor.create(shape, elements.mapIndexed { i, f ->
//            f + alpha * (t as CpuTensor).elements[i]
//        }.toFloatArray())
//    }
//
//    override fun map(transform: (Float) -> Float): Tensor {
//        val fa = elements.map(transform).toFloatArray()
//        return Tensor.create(shape, fa)
//    }
//
//    override fun equals(other: Any?): Boolean {
//        if (this === other) return true
//        if (javaClass != other?.javaClass) return false
//
//        other as CpuTensor
//
//        if (!shape.contentEquals(other.shape)) return false
//        if (!elements.contentEquals(other.elements)) return false
//        if (batch != other.batch) return false
//
//        return true
//    }
//
//    override fun hashCode(): Int {
//        var result = shape.contentHashCode()
//        result = 31 * result + elements.contentHashCode()
//        result = 31 * result + batch
//        return result
//    }
//
//    //    private data class Cut(var start: Int, var end: Int, val shape: IntArray, var size: Int) {
////        companion object {
////            fun get(s: IntArray, r: ClosedRange<Int>): Cut {
////                val xSize = r.endInclusive + 1 - r.start
////                val shape = if (xSize == 1) {
////                    s.drop(1).toIntArray()
////                } else {
////                    (arrayOf(xSize) + s.drop(1)).toIntArray()
////                }
////
////                val size = shape.dot()
////                val start = r.start * size / xSize
////                val end = (1 + r.endInclusive) * size / xSize
////                return Cut(start, end, shape, size)
////            }
////        }
////    }
//
//}
