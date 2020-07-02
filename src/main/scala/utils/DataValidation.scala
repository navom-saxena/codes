package utils

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{RangePartitioner, SparkContext}

object DataValidation extends App {

  val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
  val sc: SparkContext = spark.sparkContext

  import spark.implicits._

  val r: DataFrame = sc.parallelize(Seq(1, 2, 34, 4)).toDF()
  val r1: DataFrame = sc.parallelize(Seq(1, 3, 2, 4)).toDF()

  assertLargeDataFrameEquality(r, r1, "table")

  def performCheck(tables: Array[String], metaData: Map[String, (Array[String], Option[String])] = Map.empty): Unit = {
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
          (filterQuery.fold(actualDf) {
            actualDf.filter
          }, filterQuery.fold(expectedDf) {
            expectedDf.filter
          })
        } else {
          (spark.table(s"$hiveDb.$table"), spark.table(s"$sparkDb.$table"))
        }
        assertLargeDataFrameEquality(actualDf, expectedDf, table)
      } catch {
        case e: Exception =>
          println(s"exception thrown for table $table - $e")
          e.printStackTrace()
      }
    }
  }

  def assertLargeDataFrameEquality(actualDf: DataFrame, expectedDf: DataFrame, table: String): Unit = {
    println("Performing schema check -")
    actualDf.printSchema()
    expectedDf.printSchema()
    if (!actualDf.schema.equals(expectedDf.schema)) {
      throw new RuntimeException("schema mismatch")
    }

    println("Performing count check -")
    val actualCount: Long = actualDf.count
    val expectedCount: Long = expectedDf.count
    println(s"expected count - $expectedCount. Actual count - $actualCount")
    if (actualCount != expectedCount) {
      throw new RuntimeException("count mismatch")
    }

    println("Performing join - ")
    val mismatchDf: DataFrame = dataFramesJoin(actualDf, expectedDf, (actualCount / 1000).toInt)

    println("Initiating action - ")
    val count: Long = mismatchDf.count()
    if (count > 0) {
      println(s"---------> validation failed for $table")
      println(s"---------> count, $count")
      mismatchDf.show(count.toInt)
    } else {
      println(s"---------> Validation successful for table - $table.")
    }

  }

  def dataFramesJoin(df1: DataFrame, df2: DataFrame, numPartitions: Int): DataFrame = {
    val colNames: Array[String] = df1.columns
    val cols: Array[Column] = colNames.map(col)
    val columnsToJoin: Column = colNames.map(c => df1(c) === df2(c)).reduce(_ && _)
    val df1Repartitioned: Dataset[Row] = df1.repartitionByRange(numPartitions, cols: _*)
    val df2Repartitioned: Dataset[Row] = df2.repartitionByRange(numPartitions, cols: _*)
    val notInSpark: DataFrame = df1Repartitioned
      .join(df2Repartitioned, columnsToJoin, "left_anti")
      .withColumn("from", lit("hive"))
    val notInHive: DataFrame = df2Repartitioned
      .join(df1Repartitioned, columnsToJoin, "left_anti")
      .withColumn("from", lit("spark"))
    notInSpark.union(notInHive).persist(StorageLevel.MEMORY_AND_DISK)
  }

  def rddBasedMismatch(actualDf: DataFrame, expectedDf: DataFrame, table: String): Unit = {

    val actualCount: Long = actualDf.count()
    val expectedCount: Long = expectedDf.count()

    println("Sorting dataframes - ")

    val expectedIndexValue: RDD[(Long, Row)] = zipWithIndex(defaultSortDataset(actualDf).rdd)
    val resultIndexValue: RDD[(Long, Row)] = zipWithIndex(defaultSortDataset(expectedDf).rdd)

    val partitionerExpectedData: RangePartitioner[Long, Row] =
      new RangePartitioner(expectedCount.toInt / 50, expectedIndexValue)
    val partitionedExpectedRDD: RDD[(Long, Row)] = expectedIndexValue
      .partitionBy(partitionerExpectedData)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val partitionerResultData: RangePartitioner[Long, Row] =
      new RangePartitioner(actualCount.toInt / 50, resultIndexValue)
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

    val maxUnequalRowsToShow: Int = 100
    val count: Long = unequalRDD.count()
    if (count > 0) {
      println(s"--------> validation failed for $table")
      println(s"---------> count, ${unequalRDD.count()}")
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

}