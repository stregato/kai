package org.soft.kai.grad

import org.soft.kai.tensors.Tensor

class ScalarMulGrad(var a: Float): G1() {
    override fun `∂x`(x: Tensor) =
        x.map { a }

}

class TensorMulGrad: G2() {

    override fun `∂a`(a: Tensor, b: Tensor): Tensor {
        return b.t()
    }

    override fun `∂b`(a: Tensor, b: Tensor): Tensor {
        return a.t()
    }

}

class TensorAddGrad: G2() {

    override fun `∂a`(a: Tensor, b: Tensor): Tensor {
        return a
    }

    override fun `∂b`(a: Tensor, b: Tensor): Tensor {
        return a.t()
    }

}
