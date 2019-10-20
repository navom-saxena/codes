package scalalearning

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object SparkDataProcessing extends App {

  /*
  Write a Spark job that processes comma-seperated lines that look like the below example to pull out Key Value pairs.
  Given the following data lets say

  inputRDD:RDD[String] =

   Row-Key-001, (a,b,c),(b,c,a),(c,a,b)
   Row-Key-002, (b,c,a),(c,a,b),(a,b,c)
   Row-Key-003, (c,a,b),(a,b,c),(b,c,a)
   You'll want to create an function that contains the following data as
   patternRDD: RDD[String]=

   Row_1,(a,b)
   Row_2,(b,c)
   Row_3,(c,a)

   def takeAndProcessData(patternRDD: RDD[String]): RDD[Map(String,Map((String,String),Int))] = {
   inputRDD //("your code here")
   }
   Rows in both RDD will always be same.
   RDD[Map(String,Map((String,String),Int))]

   Map(Row-Key-001 -> Map((a,b) -> 2)),
   Map(Row-Key-002 -> Map((b,c) -> 2)),
   Map(Row-Key-003 -> Map((c,a) -> 2))

   */

  val conf: SparkConf = new SparkConf().setMaster("local[1]").setAppName("")
  val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
  val sc: SparkContext = spark.sparkContext
  val inputFile: String = "test.csv"
  //  val inputRDD: RDD[String] = sc.parallelize(List("Row-Key-001, (a,b,c),(b,c,a),(c,a,b)",
  //    "Row-Key-002, (b,c,a),(c,a,b),(a,b,c)", "Row-Key-003, (c,a,b),(a,b,c),(b,c,a)"))
  val inputRDD: RDD[String] = sc.textFile(inputFile)
  val patternRDD: RDD[String] = createPatternRDD(inputRDD)
  val processedData: RDD[Map[String, Map[(String, String), Int]]] = takeAndProcessData(patternRDD)

  def createPatternRDD(inputRDD: RDD[String]): RDD[String] = {
    inputRDD.map(_.replaceFirst(",", "_").split("_")).map { inputArr =>

      val rowArr: Array[String] = inputArr(0).split("-")
      val row: String = rowArr(0)
      val rowNo: String = rowArr(2).toInt.toString
      val rowWithNo: String = Array(row, rowNo).mkString("_")

      val alphabetArr: Array[String] = inputArr(1).replaceAll("[()]", "").split(",")
      val alphabetTuple: String = s"(${alphabetArr(0)},${alphabetArr(1)})"
      s"$rowWithNo,$alphabetTuple"
    }
  }

  def takeAndProcessData(patternRDD: RDD[String]): RDD[Map[String, Map[(String, String), Int]]] = {
    patternRDD.map { patternString =>
      val patternArr: Array[String] = patternString.replaceFirst(",", "_").split("_")
      val outerMapKey: String = s"${patternArr(0)}-00${patternArr(1)}"
      val innerMapKey: Array[String] = patternArr(2).replaceAll("[()]", "").split(",")
      Map(outerMapKey -> Map((innerMapKey(0).trim, innerMapKey(1).trim) -> innerMapKey.length))
    }
  }

  processedData.foreach(println)

}
