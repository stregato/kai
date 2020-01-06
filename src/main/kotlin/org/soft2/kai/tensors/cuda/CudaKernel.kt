package org.soft2.kai.tensors.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.*
import org.soft2.kai.tensors.Handle
import org.soft2.kai.tensors.Kernel
import java.lang.Exception

object CudaKernel: Kernel {

    var available: Boolean

    override fun allocate(floatArray: FloatArray): Handle {
        val handle = CudaHandle(floatArray.size, cublasOperation.CUBLAS_OP_T)

        JCublas.cublasAlloc(floatArray.size, Sizeof.FLOAT, handle.pointer)
        JCublas.cublasSetVector(
            floatArray.size, Sizeof.FLOAT,
            Pointer.to(floatArray), 1,
            handle.pointer, 1
        )
        return handle
    }

    override fun size(h: Handle) = (h as CudaHandle).size

    override fun toFloatArray(h: Handle): FloatArray {
        check (h is CudaHandle) { "Handle must be a CUDA Handle" }

        val dest = FloatArray(h.size)
        JCublas.cublasGetVector(
            h.size, Sizeof.FLOAT,
            h.pointer, 1,
            Pointer.to(dest), 1

        )
        return dest
    }

    override fun release(h: Handle) {
        JCublas.cublasFree( (h as CudaHandle).pointer )
    }

    override fun matrixMul(
        a: Handle,
        b: Handle,
        n: Int,
        m: Int,
        q: Int,
        beta: Float
    ): Handle {
        check(a is CudaHandle) { " a must be CUDA handle" }
        check(b is CudaHandle) { " b must be CUDA handle" }

        val batchA = a.size / n / m
        val batchB = b.size / m / q

        check(batchA == batchB || batchA == 1 || batchB == 1) {
            """CUDA batch multiplication requires batch of same size or of size 1 
                | batchA = $batchA and batchB = $batchB
            """.trimMargin()
        }

        val strideA = if (batchA == 1) 0L else n*m.toLong()
        val strideB = if (batchB == 1) 0L else m*q.toLong()

        val batchCount = maxOf(batchA, batchB)
        val strideC = m * n.toLong()

        val c = Pointer()
        JCublas.cublasAlloc(m * n * batchCount, Sizeof.FLOAT, c)
        JCublas2.cublasSgemmStridedBatched(
            cublasHandle,
            a.op, b.op,
            m, n, q,
            pOne,
            a.pointer, m,
            strideA,
            b.pointer, n,
            strideB,
            if (beta == 0f) pZero else Pointer.to(floatArrayOf(beta)),
            c, q,
            strideC,
            batchCount
        )

        return c
    }

    override fun add(a: Handle, b: Handle, alpha: Float): Handle {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun get(h: Handle, offset: Int): Float {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transpose(h: Handle, n: Int, m: Int): FloatArray {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun update(h: Handle, alpha: Float, inc: Handle): Handle {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun norm(handle: Handle): Float {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun bwMul(a: Handle, b: Handle): Handle {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun expand(a: Handle, volume: Int, targetVolume: Int): FloatArray {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }


    class CudaHandle(val size: Int, val op: Int) {
        val pointer = Pointer()
    }

    private val cublasHandle: cublasHandle
    private val pOne: Pointer = Pointer.to(floatArrayOf(1f))
    private val pZero: Pointer = Pointer.to(floatArrayOf(0f))

    init {
        try {
            JCublas.setExceptionsEnabled(true)
            JCublas2.setExceptionsEnabled(true)
        } catch (e: Exception) {
            available = false
        }

        cublasHandle = cublasHandle()
        available = JCublas2.cublasCreate(cublasHandle) == cublasStatus.CUBLAS_STATUS_SUCCESS
    }

    @JvmStatic
    fun finalize() {
        JCublas2.cublasDestroy(cublasHandle)
    }

}