package org.soft.kai

import org.soft.kai.tensors.Tensor

class Model(vararg transformations: Transformation): (Tensor) -> Tensor {

    private var ts = arrayOf(*transformations)

    override fun invoke(t: Tensor): Tensor {
        var out = t
        for (f in ts) {
            out = f(t)
        }
        return out
    }


}