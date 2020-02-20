package org.soft2.kai.tensors

import org.soft2.kai.eye as eyeFun

object Cache {

    data class FillKey(val alpha: Float, val shape: IntArray) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            other as FillKey
            return  shape.contentEquals(other.shape) && alpha == other.alpha;
        }

        override fun hashCode(): Int {
            return 31 * shape.contentHashCode() + alpha.hashCode()
        }

    }

    private fun track(t: Tensor): Tensor {
        tensorsInCache.add(t)
        return t
    }

    val tensorsInCache = HashSet<Tensor>()
    private val fillCache = mutableMapOf<FillKey, Tensor>()
    private val eyeCache = mutableMapOf<Int, Tensor>()

    fun eye(n: Int): Tensor {
        return track( eyeCache.getOrPut(n) {
            eyeFun(n)
        } )
    }

    fun fill(alpha: Float, vararg shape: Int, batch: Int = 1): Tensor {
        val size = shape.prod() * batch
        val key = FillKey(alpha, shape)
        return track(fillCache.getOrPut(key) {
            Tensor(shape, FloatArray(size) {alpha})
        })
    }

}