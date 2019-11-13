package org.soft.kai.grad

import org.soft.kai.tensors.Tensor

class MulGrad(var a: Float): G1() {
    override fun `∂x`(x: Tensor) =
        x.map { a }

}

class MatrixMulGrad: G2() {

    override fun `∂a`(a: Tensor, b: Tensor): Tensor {
        return b.transpose()
    }

    override fun `∂b`(a: Tensor, b: Tensor): Tensor {
        return a.transpose()
    }

}