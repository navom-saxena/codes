package utils

import java.io.{FileWriter, InputStream}

import play.api.libs.json.{JsValue, Json}

import scala.collection.immutable.Map
import scala.io.{BufferedSource, Source}

object FileUtils {

  def readJsonAsMap(path: String): Map[String, String] = {
    val fileContent: String = readResourceFile(path)
    val fileAsJson: JsValue = Json.parse(fileContent)
    Json.fromJson[Map[String, String]](fileAsJson).getOrElse(Map.empty[String, String])
  }

  def readResourceFile(path: String): String = {

    val stream: InputStream = getClass.getResourceAsStream(path)
    val lines: BufferedSource = scala.io.Source.fromInputStream(stream)
    try {
      lines.getLines().mkString
    }
    finally {
      lines.close()
    }
  }

  def readFile(path: String): List[String] = {
    val text: BufferedSource = Source.fromFile(path)
    text.getLines().toList
  }

  def saveFile(lines: List[String], path: String): Unit = {

    val fw: FileWriter = new FileWriter(path, false)
    try {
      lines.foreach(line => fw.write(line + "\n"))
    } finally {
      fw.close()
    }
  }

  def saveFile(content: String, path: String): Unit = {

    val fw: FileWriter = new FileWriter(path, false)
    try {
      fw.write(content)
    } finally {
      fw.close()
    }
  }

}
