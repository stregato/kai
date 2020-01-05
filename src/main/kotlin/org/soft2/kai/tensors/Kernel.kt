package org.soft2.kai.tensors

import org.soft2.kai.tensors.cpu.CpuKernel

typealias Handle = Any

interface Kernel {

    /** Allocate a tensor with provided content in memory */
    fun allocate(floatArray: FloatArray): Handle

    /** Get the total number of floats stored in the tensor */
    fun size(h: Handle): Int

    /** Return a float array representation of the tensor */
    fun toFloatArray(h: Handle): FloatArray

    /** Releast the handle */
    fun release(h: Handle)

    /** Multiply two tensors with the provided shapes */
    fun matrixMul(
        a: Handle, b: Handle,
        n: Int, m: Int, q: Int,
        transA: Boolean = false, transB: Boolean = false,
        beta: Float = 0f
    ):  Handle

    fun add(
        a: Handle, b: Handle,
        alpha: Float = 1f
    ): Handle

    fun get(h: Handle, offset: Int): Float

    fun transpose(h: Handle, n: Int, m: Int): FloatArray

    fun update(h: Handle, alpha: Float, inc: Handle): Handle

    fun norm(handle: Handle): Float

    fun bwMul(a: Handle, b: Handle): Handle

    fun expand(a: Handle, volume: Int, targetVolume: Int): FloatArray

    companion object {
        fun get() = CpuKernel()
    }

}

