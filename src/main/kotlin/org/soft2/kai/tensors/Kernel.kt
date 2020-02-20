package org.soft2.kai.tensors

import org.soft2.kai.tensors.cpu.CpuKernel
import org.soft2.kai.tensors.cuda.CudaKernel

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
    fun mul(
        a: Handle, b: Handle,
        n: Int, m: Int, q: Int,
        beta: Float = 0f
    ):  Handle

    /** Multiply two tensors with the provided shapes */
    fun mul(
        a: Handle, alpha: Float
    ):  Handle

    fun add(a: Handle, b: Handle, alpha: Float): Handle

    fun get(h: Handle, offset: Int): Float

    fun transpose(h: Handle, n: Int, m: Int): Handle

    fun update(h: Handle, alpha: Float, inc: Handle)

    fun norm0(handle: Handle): Float

    fun bwMul(a: Handle, b: Handle): Handle

    fun expand(a: Handle, volume: Int, targetVolume: Int): FloatArray

    companion object {
        var default: Kernel

         init {
             default = if (CudaKernel.available) CudaKernel else CpuKernel
         }
    }

}

