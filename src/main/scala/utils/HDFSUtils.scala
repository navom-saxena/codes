package utils

import java.io.{BufferedInputStream, IOException, OutputStreamWriter}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.immutable
import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Try

object HDFSUtils {

  val CORESITE_XML_PATH: String = "path/to/core-site.xml"
  val HDFSSITE_XML_PATH: String = "path/to/hdfs-site.xml"

  def hadoopFileSystem(): FileSystem = {
    val hdfsConf: Configuration = new Configuration()
    hdfsConf.addResource(new Path(CORESITE_XML_PATH))
    hdfsConf.addResource(new Path(HDFSSITE_XML_PATH))

    FileSystem.get(hdfsConf)
  }

  def calculateSize(inputHDFSPaths: List[String])
                   (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Long = {
    inputHDFSPaths
      .map(new org.apache.hadoop.fs.Path(_))
      .filter(validateHDFSPath)
      .map(fs.getContentSummary(_).getLength).sum
  }

  def validateHDFSPath(repoPath: Path)
                      (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Boolean = {
    fs.exists(repoPath)
  }

  def calculateSize(inputHDFSPaths: String)
                   (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Long = {
    val hdfsPath: Path = new org.apache.hadoop.fs.Path(inputHDFSPaths)
    calculateSize(hdfsPath)
  }

  def calculateSize(hdfsPath: Path)
                   (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Long = {
    if (validateHDFSPath(hdfsPath)) {
      fs.getContentSummary(hdfsPath).getLength
    } else {
      0
    }
  }

  def validateHDFSPath(repoPath: String)
                      (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Boolean = {
    if (fs.exists(new org.apache.hadoop.fs.Path(repoPath))) true else false
  }

  def readFile(filePath: String)
              (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): List[String] = {
    val hdfsPath: Path = new org.apache.hadoop.fs.Path(filePath)
    Source.fromInputStream(fs.open(hdfsPath)).getLines().toList
  }


  def readAndMap(path: String, mapper: (String) => Unit)
                (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Unit = {
    if (fs.exists(new Path(path))) {
      val inputStream: BufferedInputStream = new BufferedInputStream(fs.open(new Path(path)))
      Source.fromInputStream(inputStream).getLines().foreach(mapper)
    }
    else {
      throw new IOException("path not found")
    }
  }

  def write(filename: String, content: Iterator[String])
           (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Unit = {
    val path: Path = new Path(filename)
    val out: OutputStreamWriter = new OutputStreamWriter(fs.create(path, false))
    content.foreach(str => out.write(str + "\n"))
    out.flush()
    out.close()
  }

  def ls(path: String)
        (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): List[String] = {
    val files = fs.listFiles(new Path(path), false)
    val filenames = ListBuffer[String]()
    while (files.hasNext) filenames += files.next().getPath.toString
    filenames.toList
  }

  def rm(path: String, recursive: Boolean)
        (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Unit = {
    if (fs.exists(new Path(path))) {
      println("deleting file : " + path)
      fs.delete(new Path(path), recursive)
    }
    else {
      println("File/Directory" + path + " does not exist")
    }
  }

  def cat(path: String)
         (implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration)
  : Unit = Source.fromInputStream(fs.open(new Path(path))).getLines().foreach(println)

  def dirExists(hdfsDirectory: String)(implicit spark: SparkSession, sc: SparkContext, fs: FileSystem, hadoop: Configuration): Boolean = {
    fs.exists(new org.apache.hadoop.fs.Path(hdfsDirectory))
  }

  def copyFilesToNewPath(basePath: String, filePath: String, newPath: String)(implicit spark: SparkSession, sc: SparkContext, fs: FileSystem)
  : Unit = {
    val absolutePath: String = basePath + filePath
    val newAbsolutePath: String = newPath + filePath
    val listOFilesToLoad: immutable.Seq[String] = fs.listStatus(new Path(absolutePath))
      .map(x => x.getPath.toString)
      .filter(_.contains("part")).toList
    Try(fs.delete(new Path(newAbsolutePath), true))
    Try(fs.mkdirs(new Path(newAbsolutePath)))
    listOFilesToLoad.foreach(file => FileUtil.copy(fs, new Path(file), fs, new Path(newAbsolutePath),
      false, spark.sparkContext.hadoopConfiguration))
  }

  def mergeSmallFilesInHdfs(fullPath: String, dirName: String = "")(implicit spark: SparkSession, sc: SparkContext, fs: FileSystem)
  : Unit = {
    val absolutePath = fullPath + dirName
    val new_file_path = absolutePath + "_optimized/part-00000"
    val new_full_path = absolutePath + "_optimized"
    try {
      val bool: Boolean = FileUtil.copyMerge(fs, new Path(absolutePath), fs, new Path(new_file_path), true,
        sc.hadoopConfiguration, "")
      if (bool) fs.rename(new Path(new_full_path), new Path(absolutePath))
    }
    catch {
      case _: IOException => println("Unable to merge files for path", absolutePath)
    }
  }

}
