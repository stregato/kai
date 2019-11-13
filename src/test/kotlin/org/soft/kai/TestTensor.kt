package org.soft.kai


import org.junit.Assert.assertEquals
import org.junit.Test
import org.soft.kai.tensors.Cache


class TestTensor {


    @Test
    fun simpleMatrix() {
        val a = tensor("1 2 3", "4 5 6")

        //p = x*2+y

        assertEquals(1f, a[0, 0])
        assertEquals(2f, a[0, 1])
        assertEquals(3f, a[0, 2])
        assertEquals(5f, a[1, 1])
    }

    @Test
    fun simpleTensor3D() {
        val a = tensor(
            2, 3, 3,
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f,
            10f, 11f, 12f,
            13f, 14f, 15f,
            16f, 17f, 18f
        )


        assertEquals(1f, a[0, 0, 0])
        assertEquals(4f, a[0, 1, 0])
        assertEquals(8f, a[0, 2, 1])
        assertEquals(14f, a[1, 1, 1])
    }

    @Test
    fun mutable() {
        val a = tensor("1 3").mutable()

        a.update(-1f, Cache.fill(1f, 2, 1))
        assert(a.hasElements("0 2"))
    }

    @Test
    fun randomMatrix() {
        val a = random(10, 12)

        assert(a.shape.contentEquals(intArrayOf(10, 12)))
    }

    @Test
    fun mulByScalar() {
        val a = tensor("1 2", "3 4")
        val b = a * 3f

        assert(b.hasElements("3 6 9 12"))
    }

    @Test
    fun mulByOne() {
        val a = tensor("1 2 3", "4 5 6", "7 8 9")
        val b = eye(3)

        assert((a*b).hasElements("1 2 3 4 5 6 7 8 9"))
    }

    @Test
    fun mulByInv() {
        val a = tensor("1 2", "3 4")
        val b = tensor("-2 1", "1.5 -0.5")

        assert((a*b).hasElements("1 0 0 1"))
    }


    @Test
    fun mulByVector() {
        val a = tensor("1 2",
                       "2 3")
        var b = tensor("2 3",
                       "4 5")

        assert((a*b).hasElements("10 13 16 21"))

        b = tensor("2",
                   "4")

        assert((a*b).hasElements("10 16"))
    }

    @Test
    fun addition() {
        val a = tensor(1f, 2f, 3f, 4f)
        val b = tensor(4) { 1f }

        val s = a + b
        val d = a - b
        assert(s.hasElements(2f, 3f, 4f, 5f))
        assert(d.hasElements(0f, 1f, 2f, 3f))
    }

    @Test
    fun statistics() {
        val t = tensor(1f, 2f, 3f, 4f)
        assertEquals(10f, t.sum)
        assertEquals(2.5f, t.mean)

        val z = zeros(10)
        assertEquals(0f, z.sum)

//        val s = tensor(1f..9f)
//        assertEquals(2.581989f, s.std) TODO

    }

    @Test
    fun transpose() {
        val x = tensor("1 2 3", "4 5 6")
        val y = x.transpose()

        assertEquals(x.shape[0], y.shape[1])
        assertEquals(x.shape[1], y.shape[0])
        assert(y.hasElements("1 4 2 5 3 6"))
    }

    @Test
    fun grad() {
        val x = tensor("1 2", "3 4")
        val y = x*2f

        val gr = y.gradient?.let { g -> g() }
        assert(gr!![0].hasElements("2 2 2 2"))
    }

    @Test
    fun relu() {
        val t = tensor(-5f..5f)
        val r = t.map { if (it > 0f) it else 0f }

        assertEquals(0f, t.sum)
        assertEquals(15f, r.sum)
    }




}

