//package org.soft.kai.tensors.cuda
//
//
//import jcuda.Pointer
//import jcuda.jcublas.JCublas
//import jcuda.jcublas.cublasOperation
//import org.soft.kai.tensors.Tensor
//import org.soft.kai.tensors.cpu.CpuTensor
//import org.soft.kai.toString
//
//
//fun useCuda(): Boolean {
//    try {
//        if (CudaWrapper.cudaAvailable) {
//            Tensor.create = { s, e -> CudaTensor(s, e) }
//            return true
//        }
//    } catch (_: Throwable) {}
//
//    Tensor.create = { s, e -> CpuTensor(s, e) }
//    return false
//}
//
//
//open class CudaTensor(
//    shape: IntArray,
//    private var pointer: Pointer,
//    override var batch: Int = 1,
//    private var trans: Boolean = false
//) : CpuTensor(shape, dummyElements) {
//
//    companion object {
//        val dummyElements = FloatArray(0)
//    }
//
//    constructor(shape: IntArray, elements: FloatArray): this(shape, Pointer(), cublasOperation.CUBLAS_OP_N) {
//        CudaWrapper.copyToGpu(elements, pointer)
//        trans = true
//    }
//
//     override var elements: FloatArray = dummyElements
//         get() = toFloatArray()
//
//
//    override val norm2: Float
//        get() = CudaWrapper.nrm2(pointer, volume*batch)
//
//
//    override fun toFloatArray(): FloatArray {
//        if (elements.isEmpty()) {
//            elements = CudaWrapper.copyFromGpu(pointer, volume*batch)
//        }
//        return elements
//    }
//
//
//    override operator fun plusAssign(t: Tensor) {
//        val b = t as CudaTensor
//        check (volume*batch == b.volume*b.batch) {
//            "Incompatible tensor sizes ${toString(b.shape)} != ${toString(shape)}"
//        }
//
//        elements = dummyElements
//        CudaWrapper.addInplace(pointer, b.pointer, volume*batch)
//    }
//
//
//    override operator fun times(t: Tensor): Tensor {
//        require(shape.size < 3 && t.shape.size < 3) {
//            "Multiplication with dimension higher than 2 is not supported"
//        }
//
//        require(shape.size > 1 && shape[1] == t.shape[0]) {
//            """Incompatible tensor sizes $shape[0]*$shape[1]
//                and ${t.shape[0]}*${t.shape[1]}"""
//        }
//
//        val b = t as CudaTensor
//        val c = CudaWrapper.mul(pointer, b.pointer,
//                                shape, b.shape,
//                                batch, b.batch,
//                                trans, b.trans)
//        val n = b.shape[0]
//        val m = if (b.shape.size > 1) b.shape[1] else b.shape[0]
//        val cBatch = maxOf(batch, b.batch)
//
//        return CudaTensor(intArrayOf(n, m), c, cBatch)
//    }
//
//
//
//    override fun add(alpha: Float, t: Tensor): Tensor {
//        val b = t as CudaTensor
//        check (volume*batch == b.volume*b.batch) {
//            "Incompatible tensor sizes ${toString(b.shape)} != ${toString(shape)}"
//        }
//
//        val c = CudaWrapper.add(pointer, b.pointer,volume*batch, alpha)
//        return CudaTensor(shape, c, batch)
//    }
//
//
//    protected fun finalize() {
//        JCublas.cublasFree(pointer)
//    }
//}