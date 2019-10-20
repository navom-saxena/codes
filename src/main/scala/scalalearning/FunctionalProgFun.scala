package scalalearning

import scala.collection.immutable
import scala.collection.immutable.List
import scala.reflect.ClassTag

/* scala code to compose FunctionX instances and chain them to trigger composed function*/

object FunctionalProgFun extends App {

  implicit val emptyStr: String = ""

  implicit def optionToString[T: ClassTag](option: Option[T])(implicit emptyString: String): String = {
    option match {
      case Some(value) =>
        value match {
          case x: List[_] => x.mkString(",")
          case _ => value.toString
        }
      case None => emptyString
    }
  }

  implicit def optionToInt(option: Option[Int]): Int = {
    option match {
      case Some(value) => value
      case None => 0
    }
  }

  val add: Int => Int = (x: Int) => {
    x + 10
  }
  val sub: Int => Int = (x: Int) => {
    x - 10
  }
  val multiply: Int => Int = (x: Int) => {
    x * 10
  }
  val aAndS: Int => Int = add.compose(sub)
  val functionList: immutable.Seq[Int => Int] = List(add, sub, multiply)
  val valueList: immutable.Seq[Int] = (1 to 10).toList
  val composed: Int => Int = functionList.reduce(_.compose(_))
  val appliedList: Seq[Int] = myMap(valueList, functionList)

  def removeOption[T: ClassTag](option: Option[T], fallBackValue: T): T = {
    option match {
      case Some(value) if value.isInstanceOf[String] && value.asInstanceOf[String].nonEmpty => value
      case Some(value) if value.isInstanceOf[Int] => value
      case _ => fallBackValue
    }
  }

  def addM(a: Int): Int = {
    a + 10
  }

  def subM(a: Int): Int = {
    a - 10
  }

  def mulM(a: Int): Int = {
    a * 10
  }

  aAndS(10)
  composed(10)

  def myMap(listValue: Seq[Int], listFunction: Seq[Int => Int]): Seq[Int] = {
    listFunction.foldLeft(listValue) { case (listOfIntsAndAppliedFn, anonFn) => listOfIntsAndAppliedFn.map(anonFn) }
  }

  appliedList.foreach(println)

}
