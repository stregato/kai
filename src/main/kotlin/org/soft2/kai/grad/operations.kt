package org.soft2.kai.grad

import org.soft2.kai.tensors.Tensor

class ScalarMulGrad(var a: Float): G1() {
    override fun `∂x`(x: Tensor) =
        x.map { a }

}

class TensorMulGrad: G2() {

    override fun `∂a`(a: Tensor, b: Tensor): Tensor {
        return b.t().expand(*a.shape)
    }

    override fun `∂b`(a: Tensor, b: Tensor): Tensor {
        return a.t().expand(*b.shape)
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
