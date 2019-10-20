package scalalearning

import scala.collection.immutable.HashMap
import scala.util.Try

// testing simple scala code, creating singleton objects and more

object ScalaPractice extends App {

  val t: Int = 6
  val f: Int = 7
  var i: Int = 0
  val s = "hi"
  val a: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  t + f
  val z = List(1, 2, 3)
  val ur = Seq(1, 2, 3, 4, 5)
  val ur2 = Set(1, 2, 3, 4)
  val u: Unit = println("------------")
  while (i < a.length) {
    println(a(i))
    i = i + 1
  }
  val x: HashMap[Int, String] = new HashMap[Int, String]()
  ur.apply(6)
  val te: ChecksumAccumulator = new ChecksumAccumulator
  ur2.apply(6)

  a.exists(_ > 0)
  val x1: Rational = new Rational(1, 2)

  for (value <- a) {
    println(value)
  }
  val y1: Rational = new Rational(2, 3)

  println("-------------------")
  println {
    "Hello, world!"
  }

  a.foreach((x: Int) => println(x))
  val increase: Int => Int = (x: Int) => x + 1
  1 :: z
  z.::(2)
  z.reduceLeft((a, b) => if (a > b) b else a)
  1.->(5)


  import scala.collection.mutable

  var jetSet: Set[String] = Set("Boeing", "Airbus")

  implicit def intToRational(x: Int): Rational = new Rational(x)

  def first(x: Int): Int => Int = (y: Int) => x + y

  class ChecksumAccumulator {
    private var sum: Int = 0

    def add(b: Byte): Unit = {
      sum += b
    }

    def checksum(): Int = ~(sum & 0xFF) + 1
  }

  println(x1 + y1)

  ChecksumAccumulator
  Symbol
  Try("").getOrElse("")

  class Rational(n: Int, d: Int) {
    require(d != 0)
    private val g = gcd(n.abs, d.abs)
    val numerator: Int = n / g
    val denominator: Int = d / g

    def this(n: Int) = this(n, 1)

    def +(that: Rational): Rational =
      new Rational(
        numerator * that.denominator + that.numerator * denominator,
        denominator * that.denominator
      )

    def +(i: Int): Rational =
      new Rational(numerator + i * denominator, denominator)

    def -(that: Rational): Rational =
      new Rational(
        numerator * that.denominator - that.numerator * denominator,
        denominator * that.denominator
      )

    def -(i: Int): Rational =
      new Rational(numerator - i * denominator, denominator)

    def *(that: Rational): Rational =
      new Rational(numerator * that.numerator, denominator * that.denominator)

    def *(i: Int): Rational =
      new Rational(numerator * i, denominator)

    def /(that: Rational): Rational =
      new Rational(numerator * that.denominator, denominator * that.numerator)

    def /(i: Int): Rational =
      new Rational(numerator, denominator * i)

    override def toString: String = numerator + "/" + denominator

    @scala.annotation.tailrec
    private def gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)
  }

  object ChecksumAccumulator {
    private val cache = mutable.Map.empty[String, Int]

    def calculate(s: String): Int =
      if (cache.contains(s))
        cache(s)
      else {
        val acc: ChecksumAccumulator = new ChecksumAccumulator
        for (c <- s)
          acc.add(c.toByte)
        val cs: Int = acc.checksum()
        cache += (s -> cs)
        cs
      }
  }

}
