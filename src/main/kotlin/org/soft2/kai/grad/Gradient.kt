package org.soft2.kai.grad

import org.soft2.kai.io.TensorReader
import org.soft2.kai.tensors.*

typealias C1 = (Tensor) -> Tensor
typealias C2 = (Tensor, Tensor) -> Tensor
typealias M1 = (Tensor) -> Float

data class Step(val source: Tensor, var label: String, val derivative: C1,
                var gradient: Tensor? = null, var update: Tensor? = null) {
    override fun toString() =
        "${label}(source=${source}, gradient=${gradient}, update=${update}"
}


fun Tensor.trace(x1: Tensor, l1: String, d1: C1): Tensor {
    this.flow = arrayOf(Step(x1, l1, d1))
    return this
}

fun Tensor.trace(x1: Tensor, l1: String, d1: C1, x2: Tensor, l2: String, d2: C1): Tensor {
    this.flow = arrayOf(Step(x1, l1, d1), Step(x2, l2, d2))
    return this
}

fun Tensor.trace(x1: Tensor, l1: String, d1: C1, x2: Tensor, l2: String, d2: C1, x3: Tensor, l3: String, d3: C1): Tensor {
    this.flow = arrayOf(Step(x1, l1, d1), Step(x2, l2, d2), Step(x3, l3, d3))
    return this
}

infix fun Tensor.gradient(x: Tensor): List<Tensor?> {
    for (step in flow) {
        step.gradient = step.derivative(x)
        step.source.gradient(step.gradient!!)
    }
    return flow.map { it.gradient }
}


fun Tensor.backpropagate(e: Tensor, update: (diff: Tensor, lastUpdate: Tensor?) -> Tensor): Int {
    var numberOfUpdates = 0

    this gradient e

    for (s in this.flow) {
        if ( s.source.mutable ) {
            s.update = update(s.gradient!!, s.update)
            s.source.update(-1f, s.update!!)
            numberOfUpdates++
        }
        if ( s.source.flow.isNotEmpty() ) {
            numberOfUpdates += s.source.backpropagate(s.gradient!!, update)
        }
    }
    return numberOfUpdates
}

fun defaultUpdate(diff: Tensor, lastUpdate: Tensor?) = diff

infix fun Tensor.backpropagate(e: Tensor) = this.backpropagate(e, ::defaultUpdate)

fun pipe(vararg fs: C1) = ({ t: Tensor ->
    fs.fold(t) { acc, f ->
        f(acc)
    }
})


fun diff0(E: Tensor, Y: Tensor) = E-Y

fun exponentialDown(iteration: Int) = defaults.learningRate / (2 shl iteration)

fun learn(f: C1, X: Tensor, E: Tensor, times: Int = 1, diff: C2 = ::diff0, norm: M1 = ::norm0,
          learnRate: (Int)-> Float = ::exponentialDown): Float {
    var d = zeros(shape())
    repeat(times) {
        val Y = f(X)
        d = diff(Y,E)
        Y.backpropagate( d * learnRate(it) )
    }
    return norm(d)
}

fun learn(f: C1, xs: TensorReader, ys: TensorReader, times: Int = 1, batchSize: Int = defaults.batchSize,
          diff: C2 = ::diff0, norm: M1 = ::norm0, learnRate: (Int)-> Float = ::exponentialDown): Float {

    var error = 0f
    do {
        val X = xs.read(batchSize)
        val Y = ys.read(batchSize)
        error += learn(f, X, Y, times, diff, norm, learnRate)
    } while(!xs.eof && !ys.eof)

    return error
}
