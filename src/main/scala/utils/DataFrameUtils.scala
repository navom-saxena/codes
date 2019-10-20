package utils

import java.io.IOException

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{lit, split}
import org.apache.spark.sql.{DataFrame, Dataset, SaveMode, SparkSession}

import scala.collection.{immutable, mutable}

object DataFrameUtils {

  def readTextFile(path: String)(implicit spark: SparkSession): DataFrame = {
    spark.read.textFile(path).toDF()
  }

  def readCsvFile(path: String)(implicit spark: SparkSession): DataFrame = {
    spark.read.format(source = "csv").option("header", "true").load(path)
  }

  def checkColumnExists(inputDataFrame: DataFrame, checkColList: List[String]): DataFrame = {
    val dFCols: immutable.Seq[String] = inputDataFrame.columns.toList
    checkColList.filter(dFCols.contains).foldLeft(inputDataFrame) {
      case (dataFrame: DataFrame, col: String) => dataFrame.withColumn(col, lit(""))
    }
  }

  def sparkSplitColumnValueToMultipleColumns(inputDataFrame: DataFrame, columnToSplit: String, explodeToMultipleColumns: List[String])
  : DataFrame = {
    explodeToMultipleColumns.foldLeft(inputDataFrame) {
      case (dataFrame: DataFrame, column: String) =>
        dataFrame.withColumn(column, split(dataFrame(columnToSplit), "\t").getItem(explodeToMultipleColumns.indexOf(column)))
    }
  }

  def renameDataFrameColumns(inputDataFrame: DataFrame, inputColumnList: List[String], renameString: String): DataFrame = {
    inputColumnList.foldLeft(inputDataFrame) {
      case (dataFrame: DataFrame, column: String) => dataFrame.withColumnRenamed(column, column + renameString)
    }
  }

  def extractColumnsDataFromPath(fullPath: String, filePath: String, tableName: String, database: String,
                                 inputColumnArray: Array[String], returnDataFrame: Boolean = true,
                                 includePipeline: Boolean = false)(implicit spark: SparkSession): org.apache.spark.sql.DataFrame = {
    val filePathSplitToArray: Array[String] = filePath.toString.split('/')
    val columnsMapped: mutable.Map[String, String] = inputColumnArray.foldLeft(mutable.Map[String, String]()) {
      case (mutableMap: mutable.Map[String, String], element: String) =>
        mutableMap += (element -> filePathSplitToArray(inputColumnArray.indexOf(element)))
    }
    val allColumnPartitions: String = columnsMapped.map(key => key._1 + "='" + key._2).mkString("',")
    val columnPartition: String = allColumnPartitions.substring(0, allColumnPartitions.length - 1)
    val columnsForWhereClause = columnPartition.replaceAll(",", " and ")
    try {
      spark.sql(s"alter table $database.$tableName drop partition($columnPartition)")
    }
    catch {
      case _: Exception => println("partition does not exist previously")
    }
    if (returnDataFrame) {
      spark.sql(s"alter table $database.$tableName add partition($columnPartition) location '$fullPath'")
      spark.sql(s"select * from  $database.$tableName where $columnsForWhereClause")
    }
    else {
      spark.sql(s"alter table $database.$tableName add partition($columnPartition) location '$fullPath'")
    }
  }

  def writeDataFrameToHive(outputDF: DataFrame, createTable: Boolean = false, format: String = "text",
                           saveMode: SaveMode = SaveMode.Append, database: String = "test", table: String): Unit = {
    if (createTable) {
      outputDF.write.format(format).saveAsTable(s"$database.$table")
    } else {
      outputDF.write.format(format).mode(saveMode).insertInto(s"$database.$table")
    }
  }

  def saveDSToCsv[T](dataset: Dataset[T], csvPath: String, overWrite: Boolean = false)(implicit fs: org.apache.hadoop.fs.FileSystem)
  : Unit = {
    csvPath match {
      case dontChangePath if fs.exists(new Path(dontChangePath)) && !overWrite =>
        throw new IOException(s"$dontChangePath exists and overwrite is false")
      case _ =>
        fs.delete(new Path(csvPath), true)
        val tmpInputDir: String = csvPath + "-temp-folder"
        dataset.coalesce(1).write.option("header", "true").csv(tmpInputDir)
        val file: String = fs.globStatus(new Path(s"$tmpInputDir/part*"))(0).getPath.getName
        fs.rename(new Path(s"$tmpInputDir/" + file), new Path(csvPath))
        fs.delete(new Path(tmpInputDir), true)
    }
  }

  def saveRDDToSingleFile[T](rdd: RDD[T], stringPath: String)(implicit spark: SparkSession, fs: FileSystem): Unit = {
    val tempStringPath = stringPath + "_temp_folder"
    rdd.repartition(1).saveAsTextFile(tempStringPath)
    val filePath: Path = new Path(stringPath)
    val oldFilePath: Path = new Path(stringPath + "_old")
    if (fs.exists(filePath)) {
      if (fs.exists(oldFilePath)) {
        fs.delete(oldFilePath, true)
      }
      fs.rename(filePath, oldFilePath)
    }
    val file: String = fs.globStatus(new Path(s"$tempStringPath/part*"))(0).getPath.getName
    fs.rename(new Path(s"$tempStringPath/" + file), filePath)
    fs.delete(new Path(tempStringPath), true)
  }

}
