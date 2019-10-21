package scalalearning

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object DataFrameValidations extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()

  val t0 = spark.read.option("header", "true").csv("file:///Users/navomsaxena/Downloads/Teradat.csv")
  val f0 = spark.read.option("header", "true").csv("file:///Users/navomsaxena/Downloads/foundry.csv")

  val activeDtmz: String => String = _.replaceAll("T"," ").replaceAll("Z", "")
  import org.apache.spark.sql.functions.udf
  val activeDtmzUDF: UserDefinedFunction = udf(activeDtmz)

  val t: DataFrame = t0
    .withColumn("inactive_dtmz", trim(activeDtmzUDF(t0("inactive_dtmz"))))
    .withColumn("customer_id", trim(t0("customer_id")))
    .withColumn("SECURE_TRAVELER_SEQ_NUM", trim(t0("SECURE_TRAVELER_SEQ_NUM")).cast(IntegerType))
    .withColumn("SUP_TRAVEL_NUMBER_TYPE", trim(t0("SUP_TRAVEL_NUMBER_TYPE")))

  val f: DataFrame = f0
    .withColumn("inactive_dtmz",trim(activeDtmzUDF(f0("inactive_dtmz"))))
    .withColumn("customer_id", trim(f0("customer_id")))
    .withColumn("secure_traveler_sequence_number", trim(f0("secure_traveler_sequence_number")).cast(IntegerType))
    .withColumn("supplemental_travel_number_type", trim(f0("supplemental_travel_number_type")))

  val j: DataFrame = f.join(t,
    f("customer_id") === t("customer_id")
    && f("secure_traveler_sequence_number") === t("SECURE_TRAVELER_SEQ_NUM")
    && f("supplemental_travel_number_type") === t("SUP_TRAVEL_NUMBER_TYPE")
    && f("active_dtmz") === t("active_dtmz")
    && f("inactive_dtmz") === t("inactive_dtmz")
    && f("update_id") === t("update_id")
    , "leftanti"
  )

  j.show(false)
  println("count -----", j.count(), t.count(), f.count())

  val f2: Dataset[Row] = f.groupBy("customer_id","secure_traveler_sequence_number","supplemental_travel_number_type")
    .agg(count("*").alias("cnt")).filter("cnt > 1")

  val t2: Dataset[Row] = t.groupBy("customer_id","SECURE_TRAVELER_SEQ_NUM","SUP_TRAVEL_NUMBER_TYPE")
    .agg(count("*").alias("cnt")).filter("cnt > 1")

  val j2: DataFrame = f2.join(t2,
    f2("customer_id") === t("customer_id")
      && f2("secure_traveler_sequence_number") === t2("SECURE_TRAVELER_SEQ_NUM")
      && f2("supplemental_travel_number_type") === t2("SUP_TRAVEL_NUMBER_TYPE")
    ,"leftanti"
  )

  j2.show()
  println("count f2 t2", f2.count(), t2.count(), j2.count())

}
