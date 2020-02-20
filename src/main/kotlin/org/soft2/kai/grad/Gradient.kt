package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

typealias C1 = (Tensor) -> Tensor

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
