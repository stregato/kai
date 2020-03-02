package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

class Optimizer {

//    val weights = mutableSetOf<Te>()

    var learningRate = 0.01f

    operator fun invoke(t: Tensor) {
//        t.gradient?.let { g ->
//            val ys = g()
//
//            g.xs.forEachIndexed{ i, x ->
//                if ( weights.contains(x) ) {
//                    (x as MutableTensor).update(learningRate, ys[i] )
//                }
//
//                g.xs.forEach { invoke(it) }
//            }
//
//        }
    }
}