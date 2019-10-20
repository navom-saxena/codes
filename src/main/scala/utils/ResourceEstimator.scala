package utils

import com.typesafe.scalalogging.LazyLogging
import jdk.nashorn.internal.ir.debug.ObjectSizeCalculator

import scala.collection.{Map, mutable}

//    resource estimator for spark jobs that calculates resource allocation and executors according to input size
//    use following in code -
//    val totalDataSize = computeTotalDataSize(extractedData.toMap)
//    val resourceEstimator = new ResourceEstimator(totalDataSize)
//    val numOutputPartitions = resourceEstimator.numOutputPartitions().toInt
//    val processorSparkConfigs = resourceEstimator.defineResourceConfig()


class ResourceEstimator(inputDataSize: Long) extends LazyLogging {

  val inputDataSizeGB: Long = inputDataSize / 1024 / 1024 / 1024

  private val config: mutable.Map[String, String] = mutable.Map(
    "spark.driver.memory" -> "16G",
    "spark.executor.memory" -> "16G",
    "spark.executor.cores" -> "4",
    "spark.dynamicAllocation.enabled" -> "true",
    "spark.files.maxPartitionBytes" -> "1073741824"
  )

  private val sparkExecutorInstances: Long = 5L
  private val dynamicAllocationInitialExecutors: Long = 5L
  private val dynamicAllocationMinExecutors: Long = 5L
  private val dynamicAllocationMaxExecutors: Long = 10L


  def numOutputPartitions(): Long = math.max((inputDataSizeGB * 3), dynamicAllocationMaxExecutors)

  def defineResourceConfig(): mutable.Map[String, String] = {

    logger.info(s"Input data size in GB is ${inputDataSizeGB}")

    val (minExecutors, executorInstances, initialExecutors, maxExecutors) = if ((inputDataSizeGB > 10)) {
      val maxExecutors = (inputDataSizeGB * 3)
      val minExecutors = (inputDataSizeGB)
      val initialExecutors = ((inputDataSizeGB * 1.5).round)
      val executorInstances = (inputDataSizeGB * 2)
      (minExecutors, executorInstances, initialExecutors, maxExecutors)
    } else {
      (dynamicAllocationMinExecutors, sparkExecutorInstances, dynamicAllocationInitialExecutors, dynamicAllocationMaxExecutors)
    }

    config("spark.executor.instances") = math.max(executorInstances, sparkExecutorInstances).toString
    config("spark.dynamicAllocation.minExecutors") = math.max(minExecutors, dynamicAllocationMinExecutors).toString
    config("spark.dynamicAllocation.initialExecutors") = math.max(initialExecutors, dynamicAllocationInitialExecutors).toString
    config("spark.dynamicAllocation.maxExecutors") = math.max(maxExecutors, dynamicAllocationMaxExecutors).toString

    config
  }
}

object ResourceEstimator extends LazyLogging {

  def computeTotalDataSize(extractedData: Map[String, Map[String, List[String]]]): Long = {

    extractedData.foldLeft(0L) { case (initialSize: Long, (env: String, envData: Map[String, List[String]])) =>
      initialSize + envData.foldLeft(0L) { case (dataSize: Long, (entity: String, data: List[String])) =>
        val individualDataSize: Long = ObjectSizeCalculator.getObjectSize(data)
        logger.info(s"Object size of $entity for $env is : $dataSize")
        dataSize + individualDataSize
      }
    }
  }

}