package org.soft2.kai.tensors.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.jcublas.*
import org.soft2.kai.tensors.prod


object CudaWrapper {
    val cudaAvailable: Boolean

    private val handle: cublasHandle
    private val pOne: Pointer = Pointer.to(floatArrayOf(1f))
    private val pZero: Pointer = Pointer.to(floatArrayOf(0f))

    init {
        JCublas.setExceptionsEnabled(true)
        JCublas2.setExceptionsEnabled(true)

        handle = cublasHandle()
        cudaAvailable = JCublas2.cublasCreate(handle) == cublasStatus.CUBLAS_STATUS_SUCCESS
    }

    @JvmStatic
    fun finalize() {
        JCublas2.cublasDestroy(handle)
    }


    fun copyToGpu(source: FloatArray, dest: Pointer) {
        JCublas.cublasAlloc(source.size, Sizeof.FLOAT, dest)
        JCublas.cublasSetVector(
            source.size, Sizeof.FLOAT,
            Pointer.to(source), 1,
            dest, 1
        )
    }

    fun copyFromGpu(source: Pointer, size: Int): FloatArray {
        val dest = FloatArray(size)
        JCublas.cublasGetVector(
            size, Sizeof.FLOAT,
            source, 1,
            Pointer.to(dest), 1

        )
        return dest
    }

    fun nrm2(source: Pointer, size: Int): Float {
        val result = floatArrayOf(0f)
        JCublas2.cublasSnrm2(
            handle,
            size, source,
            1, Pointer.to(result)
        )
        return result[0]
    }

    fun mul(
        a: Pointer, b: Pointer,
        shapeA: IntArray, shapeB: IntArray,
        batchA: Int = 1, batchB: Int = 1,
        transA: Boolean = false, transB: Boolean = false,
        beta: Float = 0f
    ): Pointer {

        check(batchA == batchB || batchA == 1 || batchB == 1) {
            """CUDA batch multiplication requires batch of same size or of size 1 
                | batchA = $batchA and batchB = $batchB
            """.trimMargin()
        }

        val ta = if (transA) cublasOperation.CUBLAS_OP_T
        else cublasOperation.CUBLAS_OP_N
        val tb = if (transB) cublasOperation.CUBLAS_OP_T
        else cublasOperation.CUBLAS_OP_N

        val m = shapeA[0]
        val n = shapeB[1]
        val k = shapeA[1]
        val lda = shapeA[0]
        val ldb = shapeB[1]
        val ldc = shapeA[1]
        val strideA = if (batchA == 1) 0L else shapeA.prod().toLong()
        val strideB = if (batchB == 1) 0L else shapeB.prod().toLong()

        val batchCount = maxOf(batchA, batchB)
        val strideC = m * n.toLong()

        val c = Pointer()
        JCublas.cublasAlloc(m * n * batchCount, Sizeof.FLOAT, c)
        JCublas2.cublasSgemmStridedBatched(
            handle,
            ta, tb,
            m, n, k,
            pOne,
            a, lda,
            strideA,
            b, ldb,
            strideB,
            if (beta == 0f) pZero else Pointer.to(floatArrayOf(beta)),
            c, ldc,
            strideC,
            batchCount
        )

        return c
    }


    fun add(
        a: Pointer, b: Pointer, size: Int,
        alpha: Float = 1f
    ): Pointer {

        val c = Pointer()
        JCublas.cublasAlloc(size, Sizeof.FLOAT, c)
        JCublas.cublasScopy(size, a, 1, c, 1)
        JCublas.cublasSaxpy(size, alpha, b, 1, c, 1)
        return c
    }

    fun addInplace(
        a: Pointer, b: Pointer, size: Int,
        alpha: Float = 1f
    ) {
        JCublas.cublasSaxpy(size, alpha, b, 1, a, 1)
    }


}