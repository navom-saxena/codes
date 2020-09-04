package utils

import java.util

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataTypes, DecimalType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{RangePartitioner, SparkContext}

object DataValidation extends App {

  val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
  val sc: SparkContext = spark.sparkContext
  import spark.implicits._

  val r: DataFrame = sc.parallelize(Seq(1, 2, 34, 4)).toDF()
  val r1: DataFrame = sc.parallelize(Seq(1, 3, 2, 4)).toDF()

  assertLargeDataFrameEquality(r, r1, "table")

//  assertLargeDataFrameEquality(r, r1, "t")
//  performCheck(Array("table"), mode = DfMode)

  def performCheck(tables: Array[String],
                   metaData: Map[String, (Array[String], Option[String])] = Map.empty,
                   mode: RunMode = Both): Unit = {
    val hiveDb: String = "fdm_presentation"
    val sparkDb: String = "fdm_presentation_spark"
    tables.foreach { table =>
      try {
        val (actualDf, expectedDf) = if (metaData.contains(table)) {
          val (ignoreColumns, filterQuery): (Array[String], Option[String]) = metaData(table)
          val actualDf: DataFrame = ignoreColumns.foldLeft(spark.table(s"$hiveDb.$table")) {
            case (df, column) => df.drop(column)
          }
          val expectedDf: DataFrame = ignoreColumns.foldLeft(spark.table(s"$sparkDb.$table")) {
            case (df, column) => df.drop(column)
          }
          (filterQuery.fold(actualDf) {actualDf.filter}, filterQuery.fold(expectedDf) {expectedDf.filter})
        } else {
          (spark.table(s"$hiveDb.$table"), spark.table(s"$sparkDb.$table"))
        }
        mode match {
          case DfMode => assertLargeDataFrameEquality(actualDf, expectedDf, table)
          case RddMode => assertRddBasedEquality(actualDf, expectedDf, table)
          case Both =>
            assertLargeDataFrameEquality(actualDf, expectedDf, table)
            assertRddBasedEquality(actualDf, expectedDf, table)
        }
      } catch {
        case e: Exception =>
          println(s"exception thrown for table $table - $e")
          e.printStackTrace()
      }
    }
  }

  def assertLargeDataFrameEquality(actualDf: DataFrame, expectedDf: DataFrame, table: String): Unit = {

    println(s"Performing schema check for table $table-")

    actualDf.printSchema()
    expectedDf.printSchema()
    if (!actualDf.schema.equals(expectedDf.schema)) {
      if (!actualDf.schema.map(_.name.toLowerCase).equals(expectedDf.schema.map(_.name.toLowerCase))) {
        throw new RuntimeException("schema mismatch due to name issue")
      } else if (!actualDf.schema.map(_.dataType).equals(expectedDf.schema.map(_.dataType))) {
        throw new RuntimeException("schema mismatch due to datatype issue")
      } else if (!actualDf.schema.map(_.metadata).equals(expectedDf.schema.map(_.metadata))) {
        throw new RuntimeException("schema mismatch due to metadata issue")
      }
    }
    println(s"schema check for table $table successful")

    println(s"Performing count check for $table")
    val actualCount: Long = actualDf.count
    val expectedCount: Long = expectedDf.count
    println(s"expected count - $expectedCount. Actual count - $actualCount")
    if (actualCount != expectedCount) {
      throw new RuntimeException("count mismatch")
    }
    println(s"count check for table $table successful")

    println("Performing join - ")
    val numPartitions: Int = if ((actualCount / 1000).toInt > 0) (actualCount / 1000).toInt else 100
    val mismatchDf: DataFrame = dataFramesJoin(actualDf, expectedDf, numPartitions)

    println("Initiating action - ")
    val count: Long = mismatchDf.count()
    if (count > 0) {
      println(s"---------> validation failed for $table")
      println(s"---------> count, $count")
      mismatchDf.show()
//      mismatchDf.write.csv(s"/tmp/$table-${Instant.now().getEpochSecond}")
    } else {
      println(s"---------> Validation successful for table - $table.")
    }

  }

  def replaceNulls(df: DataFrame): DataFrame = {
    val stringDefault: String = "null-value"
    val defaultValues: Map[String, String] = Map(
      DataTypes.DateType.toString -> "1990-09-19",
      DataTypes.TimestampType.toString -> "1990-09-19 00:00:00",
      DataTypes.BooleanType.toString -> "false",
      DecimalType(10,0).toString -> "0.0",
      DecimalType(22,7).toString -> "0.0"
    )
    df.dtypes
      .filter(dType =>
        Array(DataTypes.DateType.toString, DataTypes.TimestampType.toString, DataTypes.BooleanType.toString,
          DecimalType(10,0).toString, DecimalType(22,7).toString)
          .contains(dType._2))
      .foldLeft(df) { case (accumulatedDf, dType) =>
        accumulatedDf.withColumn(dType._1,
          when(col(dType._1).isNull, defaultValues(dType._2)).otherwise(col(dType._1)))
      }
      .na.fill(stringDefault)
      .na.fill(Long.MinValue)
      .na.fill(Double.MinValue)
      .na.fill(Map.empty[String,Any])
      .na.fill(new util.HashMap[String,Any]())
  }

  def dataFramesJoin(df1: DataFrame, df2: DataFrame, numPartitions: Int): DataFrame = {
    val df1WithNullsHandled: DataFrame = replaceNulls(df1)
    val df2WithNullsHandled: DataFrame = replaceNulls(df2)
    val colNames: Array[String] = df1WithNullsHandled.columns
    val cols: Array[Column] = colNames.map(col)
    val columnsToJoin: Column = colNames.map(c => df1WithNullsHandled(c) === df2WithNullsHandled(c)).reduce(_ && _)
    val df1Repartitioned: Dataset[Row] = df1WithNullsHandled.repartitionByRange(numPartitions, cols: _*)
    val df2Repartitioned: Dataset[Row] = df2WithNullsHandled.repartitionByRange(numPartitions, cols: _*)
    val notInSpark: DataFrame = df1Repartitioned
      .join(df2Repartitioned, columnsToJoin, "left_anti")
      .withColumn("from", lit("hive"))
    val notInHive: DataFrame = df2Repartitioned
      .join(df1Repartitioned, columnsToJoin, "left_anti")
      .withColumn("from", lit("spark"))
    notInSpark.union(notInHive).persist(StorageLevel.MEMORY_AND_DISK)
  }

  def assertRddBasedEquality(actualDf: DataFrame, expectedDf: DataFrame, table: String): Unit = {

    println(s"Performing schema check for table $table")
    actualDf.printSchema()
    expectedDf.printSchema()
    if (!actualDf.schema.equals(expectedDf.schema)) {
      if (!actualDf.schema.map(_.name.toLowerCase).equals(expectedDf.schema.map(_.name.toLowerCase))) {
        throw new RuntimeException("schema mismatch due to name issue")
      } else if (!actualDf.schema.map(_.dataType).equals(expectedDf.schema.map(_.dataType))) {
        throw new RuntimeException("schema mismatch due to datatype issue")
      } else if (!actualDf.schema.map(_.metadata).equals(expectedDf.schema.map(_.metadata))) {
        throw new RuntimeException("schema mismatch due to metadata issue")
      }
    }
    println(s"schema check for table $table successful")

    println(s"Performing count check for $table")
    val actualCount: Long = actualDf.count
    val expectedCount: Long = expectedDf.count
    println(s"expected count - $expectedCount. Actual count - $actualCount")
    if (actualCount != expectedCount) {
      throw new RuntimeException("count mismatch")
    }
    println(s"count check for table $table successful")

    println("Sorting dataframes - ")

    val expectedIndexValue: RDD[(Long, Row)] = zipWithIndex(defaultSortDataset(actualDf).rdd)
    val resultIndexValue: RDD[(Long, Row)] = zipWithIndex(defaultSortDataset(expectedDf).rdd)

    val numPartitions: Int = if (expectedCount.toInt > 1000) expectedCount.toInt / 1000 else 100
    val partitionerExpectedData: RangePartitioner[Long, Row] =
      new RangePartitioner(numPartitions, expectedIndexValue)
    val partitionedExpectedRDD: RDD[(Long, Row)] = expectedIndexValue
      .partitionBy(partitionerExpectedData)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val partitionerResultData: RangePartitioner[Long, Row] =
      new RangePartitioner(numPartitions, resultIndexValue)
    val partitionedResultRDD: RDD[(Long, Row)] = resultIndexValue
      .partitionBy(partitionerResultData)
      .persist(StorageLevel.MEMORY_AND_DISK)

    println("Performing join - ")

    val unequalRDD: RDD[(Long, (Row, Row))] = partitionedExpectedRDD
      .join(partitionedResultRDD)
      .filter {
        case (_, (r1, r2)) =>
          !(r1.equals(r2) || RowComparer.areRowsEqual(r1, r2, 0.0))
      }

    val maxUnequalRowsToShow: Int = 20
    val count: Long = unequalRDD.count()
    if (count > 0) {
      println(s"--------> validation failed for $table")
      println(s"---------> count, ${unequalRDD.count()}. Showing 20 rows")
      unequalRDD.take(maxUnequalRowsToShow).foreach(println("Unequal rows - ", _))
    } else {
      println(s"---------> Validation successful for table - $table.")
    }
  }

  def zipWithIndex[T](rdd: RDD[T]): RDD[(Long, T)] = {
    rdd.zipWithIndex().map {
      case (row, idx) => (idx, row)
    }
  }

  def defaultSortDataset(df: DataFrame): DataFrame = {
    val colNames: Array[String] = df.columns.sorted
    val cols: Array[Column] = colNames.map(col)
    df.repartition(cols: _*).sortWithinPartitions(cols: _*)
  }

  trait RunMode

  case object DfMode extends RunMode

  case object RddMode extends RunMode

  case object Both extends RunMode

}