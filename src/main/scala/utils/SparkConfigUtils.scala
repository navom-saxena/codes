package utils

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import scala.collection.immutable.Map

object SparkConfigUtils extends Enumeration {

//  to set Info level to off
//  import org.apache.log4j.Logger
//  import org.apache.log4j.Level
//
//  Logger.getLogger("org").setLevel(Level.OFF)
//  Logger.getLogger("akka").setLevel(Level.OFF)

  type RunMode = Value
  val PRODUCTION, UNIT_TEST = Value

  val runMode: SparkConfigUtils.Value = PRODUCTION

  val productionSparkConfigurations: Map[String, String] = Map[String, String](
    "spark.scheduler.listenerbus.eventqueue.size" -> "300000",
    "spark.network.timeout" -> "600",
    "spark.master" -> "yarn",
    "spark.eventLog.enabled" -> "true",
    "spark.yarn.am.waitTime" -> "100000",
    "spark.yarn.executor.memoryOverhead" -> "2000",
    "spark.yarn.max.executor.failures" -> "2000",
    "sun.io.serialization.extendedDebugInfo" -> "true",
    "spark.executor.extraJavaOptions" -> "-XX ->-UseSplitVerifier",
    "spark.shuffle.consolidateFiles" -> "true",
    "spark.shuffle.compress" -> "true",
    "spark.shuffle.spill.compress" -> "true",
    "spark.dynamicAllocation.enabled" -> "false",
    "spark.executor.instances" -> "300",
    "hive.exec.dynamic.partition" -> "true",
    "hive.exec.dynamic.partition.mode" -> "nonstrict",
    "hive.exec.stagingdir" -> "/tmp/spark-hive-staging/",
    "spark.hive.mapred.supports.subdirectories" -> "true",
    "spark.hadoop.mapreduce.input.fileinputformat.input.dir.recursive" -> "true",
    "mapreduce.input.fileinputformat.input.dir.recursive" -> "true"
  )

  val unitTestSparkConfigurations: Map[String, String] = Map[String, String](
    "spark.master" -> "local[4]",
    "spark.executor.memory" -> "4g",
    "spark.app.name" -> UNIT_TEST.toString,
    "spark.sql.catalogImplementation" -> "in-memory",
    "spark.sql.shuffle.partitions" -> "1",
    "spark.sql.warehouse.dir" -> "target/spark-warehouse")

  def sparkSessionBuilder(appName: String, runMode: RunMode = PRODUCTION): SparkSession = {
    val spark = runMode match {
      case UNIT_TEST =>
        SparkSession
          .builder()
          .config(sparkConfBuilder(runMode, appName))
          .getOrCreate()
      case PRODUCTION =>
        SparkSession
          .builder()
          .enableHiveSupport()
          .config(sparkConfBuilder(runMode, appName))
          .getOrCreate()
    }
    hadoopConfigurations(spark)
  }

  def sparkConfBuilder(runMode: RunMode, appName: String = "DefaultDataExtractor"): SparkConf = {
    runMode match {
      case PRODUCTION => SparkConf(productionSparkConfigurations)
      case UNIT_TEST => SparkConf(unitTestSparkConfigurations)
    }
  }

  def SparkConf(sparkConfigurations: Map[String, String]): SparkConf = {
    new SparkConf().setAll(sparkConfigurations)
  }

  private def hadoopConfigurations(spark: SparkSession): SparkSession = {
    spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive", "true")
    spark
  }

}
