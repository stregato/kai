package org.soft2.kai


import org.junit.Assert.assertEquals
import org.junit.Test
import org.soft2.kai.tensors.*


class TestTensor {


    @Test
    fun simpleMatrix() {
        val a = matrix("1 2 3", "4 5 6")

        //p = x*2+y

        assertEquals(1f, a[0, 0])
        assertEquals(2f, a[0, 1])
        assertEquals(3f, a[0, 2])
        assertEquals(5f, a[1, 1])
    }

    @Test
    fun simpleTensor3D() {
        val a = tensor(
            shape(2, 3, 3),
            floatArrayOf(
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f,
                10f, 11f, 12f,
                13f, 14f, 15f,
                16f, 17f, 18f
            )
        )

        assertEquals(1f, a[0, 0, 0])
        assertEquals(4f, a[0, 1, 0])
        assertEquals(8f, a[0, 2, 1])
        assertEquals(14f, a[1, 1, 1])
    }

    @Test
    fun mutable() {
        val a = matrix("1 3")
        a.mutable = true

        a.update(-1f, fill(1f, intArrayOf(2)))
        assert(a.hasElements(0, 2))
    }

    @Test
    fun randomMatrix() {
        val a = random(shape(10, 12))

        assert(a.shape.contentEquals(intArrayOf(10, 12)))
    }

    @Test
    fun mulByScalar() {
        val a = matrix("1 2", "3 4")
        val b = a * 3f

        assert(b.hasElements(3, 6, 9, 12))
    }

    @Test
    fun mulByOne() {
        val a = matrix("1 2 3", "4 5 6", "7 8 9")
        val b = eye(3)

        assert((a*b).hasElements(1, 2, 3, 4, 5, 6, 7, 8, 9))
    }

    @Test
    fun mulByInv() {
        val a = matrix("1 2", "3 4")
        val b = matrix("-2 1", "1.5 -0.5")

        assert((a*b).hasElements(1, 0, 0, 1))
    }


    @Test
    fun mulByVector() {
        val a = matrix(
            "1 2",
            "2 3"
        )
        var b = matrix(
            "2 3",
            "4 5"
        )

        assert((a*b).hasElements(10, 13, 16, 21))

        b = matrix(
            "2",
            "4"
        )

        assert((a*b).hasElements(10, 16))
    }

    @Test
    fun addition() {
        val a = vector(1f, 2f, 3f, 4f)
        val b = vector(1f, 1f, 1f, 1f)

        val s = a + b
        val d = a - b
        assert(s.hasElements(2f, 3f, 4f, 5f))
        assert(d.hasElements(0f, 1f, 2f, 3f))
    }

    @Test
    fun statistics() {
        val t = vector(1f, 2f, 3f, 4f)
        assertEquals(10f, sum(t))
        assertEquals(2.5f, mean(t))

        val z = zeros(shape(10))
        assertEquals(0f, sum(z))

//        val s = tensor(1f..9f)
//        assertEquals(2.581989f, s.std) TODO

    }

    @Test
    fun transpose() {
        val x = matrix("1 2 3", "4 5 6")
        val y = x.t()

        assertEquals(x.shape[0], y.shape[1])
        assertEquals(x.shape[1], y.shape[0])
        assert(y.hasElements(1, 4, 2, 5, 3, 6))
    }

//    @Test
//    fun expand() {
//        val x = tensor("1 2 3")
//        val y = x.expand(2,3)
//
//        assert(y.hasElements(1, 2, 3, 1, 2, 3))
//
//        val a = y.reshape(2, 1)
//        val b = a.expand(2, 4)
//
//        assert(b.hasElements(1, 2, 3, 1, 2, 3))
//    }


    @Test
    fun grad() {
        val x = matrix("1 2", "3 4")
        val y = x*2f

//        val gr = y.gradient?.let { g -> g() }
//        assert(gr!![0].hasElements("2 2 2 2"))
    }

    @Test
    fun relu() {
        val t = tensor(intArrayOf(11)) { (it-5).toFloat() }
        val r = t.map { if (it > 0f) it else 0f }

        assertEquals(0f, sum(t))
        assertEquals(15f, sum(r))
    }




}

