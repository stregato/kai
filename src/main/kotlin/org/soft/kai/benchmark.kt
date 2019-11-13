package org.soft.kai

fun<R> benchmark(s: String, f: () -> R ): R {
    val start = System.currentTimeMillis()
    val r = f()
    val end = System.currentTimeMillis()
    println("bm for '$s': ${end-start}ms")
    return r
}