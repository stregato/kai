package org.soft2.kai.tensors.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.*
import org.soft2.kai.tensors.Handle
import org.soft2.kai.tensors.Kernel

object CudaKernel: Kernel {

    var available: Boolean

    override fun allocate(floatArray: FloatArray): Handle {
        val handle = CudaHandle(floatArray.size)

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

    private fun transpose(fs: FloatArray, n: Int, m: Int): FloatArray {
        val batch = fs.size / m / n
        if ( m == n) {
            for (b in 0 until batch) {
                val off = b * m * n
                for (i in 0 until n) {
                    for (j in i+1 until n) {
                        val t = fs[off + i*m + j]
                        fs[off + i*m + j] = fs[off + j*n + i]
                        fs[off + j*n + i] = t
                    }
                }
            }
            return fs
        } else {
            val ts = FloatArray(fs.size)
            for (b in 0 until batch) {
                val off = b * m * n
                for (i in 0 until m) {
                    for (j in 0 until n) {
                        ts[off + j *m + i] = fs[off + i * n + j]
                    }
                }
            }
            return ts
        }
    }

    override fun release(h: Handle) {
        JCublas.cublasFree( (h as CudaHandle).pointer )
    }

    override fun mul(
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
        val strideC = m * q.toLong()

        val p = Pointer()
        JCublas.cublasAlloc(n * q * batchCount, Sizeof.FLOAT, p)
        JCublas2.cublasSgemmStridedBatched(
            cublasHandle,
            cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
            q, n, m,
            pOne, b.pointer, q, strideB,
            a.pointer, m, strideA, if (beta == 0f) pZero else Pointer.to(floatArrayOf(beta)),
            p, q,
            strideC,
            batchCount
        )

        return CudaHandle(n * q * batchCount, p, m)
    }

    override fun mul(a: Handle, alpha: Float):  Handle {
        check(a is CudaHandle) { " a must be CUDA handle" }

        val p = Pointer()
        JCublas.cublasAlloc(a.size, Sizeof.FLOAT, p)
        JCublas.cublasScopy(a.size, a.pointer, 1, p, 1)
        JCublas.cublasSscal(a.size, alpha, p, 1)
        return CudaHandle(a.size, p)
    }


    override fun add(a: Handle, b: Handle, alpha: Float): Handle {

        check(a is CudaHandle) { " a must be CUDA handle" }
        check(b is CudaHandle) { " b must be CUDA handle" }

        val size = maxOf(a.size, b.size)
        val batchA = size / a.size
        val batchB = size / b.size

        check(size % a.size == 0) { "incompatible size ${a.size} and ${b.size}"}
        check(size % b.size == 0) { "incompatible size ${a.size} and ${b.size}"}

        val p = Pointer()
        JCublas.cublasAlloc(size, Sizeof.FLOAT, p)

        if ( batchA >= batchB ) {
            for (i in 0 until batchA) {
                JCublas.cublasScopy(a.size, a.pointer, 1, p.withByteOffset(i.toLong() * a.size * Sizeof.FLOAT), 1)
            }
            JCublas.cublasSaxpy(size, alpha, b.pointer, 1, p, 1)
        } else {
            for (i in 0 until batchB) {
                JCublas.cublasScopy(b.size, b.pointer, 1, p.withByteOffset(i.toLong() * b.size * Sizeof.FLOAT), 1)
            }
            JCublas.cublasSscal(size, alpha, p, 1)
            JCublas.cublasSaxpy(size, 1f, a.pointer, 1, p, 1)
        }

        return CudaHandle(a.size, p)
    }

    override fun get(h: Handle, offset: Int): Float {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

    override fun transpose(a: Handle, n: Int, m: Int): Handle {

        check(a is CudaHandle) { " a must be CUDA handle" }

        val p = Pointer()
        JCublas.cublasAlloc(a.size, Sizeof.FLOAT, p)
        JCublas2.cublasSgeam(cublasHandle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,
                             m, n,
                             pOne, a.pointer, n,
                             pZero, a.pointer, n,
                             p, n)

        return CudaHandle(a.size, p)
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


    class CudaHandle(val size: Int, val pointer: Pointer = Pointer(), val n: Int = 0)

    private var cublasHandle: cublasHandle = cublasHandle()
    private val pOne: Pointer = Pointer.to(floatArrayOf(1f))
    private val pZero: Pointer = Pointer.to(floatArrayOf(0f))

    init {
        try {
            JCublas.setExceptionsEnabled(true)
            JCublas2.setExceptionsEnabled(true)
            available = JCublas2.cublasCreate(cublasHandle) == cublasStatus.CUBLAS_STATUS_SUCCESS
        } catch (e: Throwable) {
            available = false
            println("Java Library Path"+System.getProperty("java.library.path"))
            print("CUDA not available. ")
            println(e.message)
        }

    }

    @JvmStatic
    fun finalize() {
        JCublas2.cublasDestroy(cublasHandle)
    }

}