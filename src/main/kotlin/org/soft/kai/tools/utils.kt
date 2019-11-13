package org.soft.kai

import java.io.File
import java.io.FileOutputStream
import java.nio.channels.Channels
import java.util.zip.GZIPInputStream

fun toString(shape: IntArray) =  "(" + shape.joinToString() + ")"

fun getFile(filename: String): File {
    val home = File(System.getProperty("user.home"))
    val folder = File(home, "./.kai")
    folder.mkdirs()

    return File(folder, filename)
}



fun downloadFile(url: String, filename: String = "", gunzip: Boolean =false): File {
    val u = java.net.URL(url)

    val file = getFile(if (filename == "") u.file else filename)
    if ( !file.exists() ) {
        val inputStream = if (gunzip) GZIPInputStream(u.openStream()) else u.openStream()
        val readableByteChannel = Channels.newChannel(inputStream);
        file.parentFile.mkdirs()
        val fileOutputStream = FileOutputStream(file);

        fileOutputStream.use {
            it.channel.transferFrom(readableByteChannel, 0 , Long.MAX_VALUE)
        }
        inputStream.close()
    }

    return file
}


