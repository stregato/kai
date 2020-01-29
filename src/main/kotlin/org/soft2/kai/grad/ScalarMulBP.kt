package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

class ScalarMulGrad(val a: Float): BackPropagation {
    override fun invoke(p1: Tensor): Tensor {
        TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
    }

}
    override operator fun invoke(t: Tensor): Tensor {
        return t * a
    }


fun scalarMulGrad(val a: Float, x: Tensor) {
    x.gradient = Gradient(x)
}







