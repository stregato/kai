//package org.soft.kai.tensors
//
//
///**
// * Matrix that supports basic operations
// */
//
//
//
//interface Tensor2 {
//
//    companion object {
//        lateinit var create: (IntArray, FloatArray) -> Tensor
//    }
//
//    /** when greater than 1, the tensor is actually a batch of defined shape */
//    val batch: Int
//
//    /** shape of the tensor in row-first notation */
//    val shape: IntArray
//
//    val mean: Float
//
//    val sum: Float
//
//    val std: Float
//
//    val scalar: Float
//
//    val norm2: Float
//
//    fun hasElements(vararg e: Float): Boolean
//
//    fun hasElements(s: String) =
//
//    fun toFloatArray(): FloatArray
//
//    fun reshape(vararg s: Int): Tensor
//
//    operator fun unaryMinus(): Tensor
//
//    operator fun plus(t: Tensor): Tensor
//
//    operator fun minus(t: Tensor): Tensor
//
//    operator fun plus(f: Float): Tensor
//
//    operator fun minus(f: Float): Tensor
//
//    operator fun plusAssign(t: Tensor)
//
//    operator fun times(t: Tensor): Tensor
//
//    operator fun times(s: Float): Tensor
//
//    operator fun div(s: Float): Tensor
//
//    override fun toString(): String
//
//    operator fun get(vararg x: Int): Float
//
//    fun map(transform: (Float) -> Float): Tensor
//
//}
//
//
//
///**
// * Returns the sum of all elements in the array.
// */
//@kotlin.jvm.JvmName("sumOfInt")
//fun IntArray.dot(): Int {
//    var dot = 1
//    for (element in this) {
//        dot *= element
//    }
//    return dot
//}
//
