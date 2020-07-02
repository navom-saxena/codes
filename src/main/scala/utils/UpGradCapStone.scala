package utils

import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object UpGradCapStone extends App {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

  def generateRevenue(order_item_quantity: String, product_price: String): Double = {
    order_item_quantity.toDouble * product_price.toDouble
  }

  val spark: SparkSession = SparkSession.builder().master("local[*]").appName("CapStoneProgram1").getOrCreate()
  val sc: SparkContext = spark.sparkContext

  val orderItems: DataFrame = spark.read.option("header", "true").csv("file:///Users/navomsaxena/Downloads/retail_store_data/order_items.csv")
  val payments: DataFrame = spark.read.option("header", "true").csv("file:///Users/navomsaxena/Downloads/retail_store_data/payments.csv")
  val products: DataFrame = spark.read.option("header", "true").csv("file:///Users/navomsaxena/Downloads/retail_store_data/products.csv")

  orderItems.show()
  payments.show()
  products.show()

  println(orderItems.count())
  println(payments.count())
  println(products.count())

  val orderPayment: DataFrame = orderItems.join(payments, orderItems("order_item_order_id") === payments("order_id")).drop(payments("order_id"))
  val orderPaymentProduct: DataFrame = orderPayment.join(products, orderPayment("order_item_product_id") === products("product_id")).drop(products("product_id"))

  orderPaymentProduct.show()

  println("order - ", orderItems.count())
  println("payments - ", payments.count())
  println("products - ", products.count())
  println("orderPayment - ", orderPayment.count())
  println("all3 - ", orderPaymentProduct.count())

  import org.apache.spark.sql.functions._
  def revenue: UserDefinedFunction = udf((order_item_quantity: String, product_price: String) => order_item_quantity.toDouble * product_price.toDouble )
  val withRevenue: Dataset[Row] = orderPaymentProduct
    .withColumn("revenue", revenue(orderPaymentProduct("order_item_quantity"),orderPaymentProduct("product_price")))


  val first = withRevenue.orderBy(desc("revenue")).filter(orderPaymentProduct("status") === "COMPLETE").limit(5)


  val second = orderPaymentProduct.filter(orderPaymentProduct("status") === "SUSPECTED_FRAUD")
    .select("order_item_order_id","revenue").groupBy("order_item_order_id").agg(sum("revenue"))

  val third = orderPaymentProduct.filter(orderPaymentProduct("status") === "COMPLETE")
    .orderBy(desc("order_item_quantity"), desc("revenue"))

  payments.distinct().show()
  first.show()
  second.show()
  third.show()
  
}
