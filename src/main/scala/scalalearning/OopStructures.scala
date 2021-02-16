package scalalearning

import scala.collection.immutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future

object OopStructures extends App {

  val f: Future[Int] = Future {
    throw new Exception
  }
  val f1: Future[Int] = f.map(x => x * 2)
  val f2: Future[Int] = f.flatMap(x => Future(x * 2))

  val x: Int = 1 + 1

  var e: Int = 3
  e = 6

  println {
    val x: Int = 1 + 1
    x + 1
  }


  val addOne: Int => Int = (x: Int) => x + 1
  val getTheAnswer: () => Int = () => 42


  class Greeter(prefix: String, suffix: String) {
    def greet(name: String): Unit =
      println(prefix + name + suffix)
  }

  val greeter: Greeter = new Greeter("Hello, ", "!")
  greeter.greet("Scala developer")

  case class CPoint(x: Int, y: Int)

  val point: CPoint = CPoint(1, 2)
  val anotherPoint: CPoint = CPoint(1, 2)
  val yetAnotherPoint: CPoint = CPoint(2, 2)


  point == anotherPoint


  object IdFactory {
    var counter = 0

    def create(): Int = {
      counter += 1
      counter
    }
  }

  (1 to 100).map(_ => Future(IdFactory.create()))

  println("hiiiii")
  println(IdFactory.counter)


  val list: List[Any] = List(
    "a string",
    732, // an integer
    'c', // a character
    true, // a boolean value
    () => "an anonymous function returning a string"
  )

  list.foreach(element => println(element))

  val s: Null = null
  val n: Option[Nothing] = None


  throw new Exception()


  class Point(var x: Int, var y: Int) {

    def move(dx: Int, dy: Int): Unit = {
      x = x + dx
      y = y + dy
    }

    override def toString: String =
      s"($x, $y)"
  }

  val point1: Point = new Point(2, 3)
  println(point1.x)


  trait Iterator[A] {
    def hasNext: Boolean

    def next(): A
  }

  class IntIterator(to: Int) extends Iterator[Int] {
    private var current: Int = 0

    override def hasNext: Boolean = current < to

    override def next(): Int = {
      if (hasNext) {
        val t: Int = current
        current += 1
        t
      } else 0
    }
  }


  val iterator: IntIterator = new IntIterator(10)
  iterator.next() // returns 0
  iterator.next() // returns 1

  abstract class A {
    val message: String
  }

  class B extends A {
    val message = "I'm an instance of class B"
  }

  trait C extends A {
    def loudMessage: String = message.toUpperCase()
  }

  class D extends B with C

  val d: D = new D

  println(d.message) // I'm an instance of class B
  println(d.loudMessage) // I'M AN INSTANCE OF CLASS B


  abstract class AbsIterator {
    type T

    def hasNext: Boolean

    def next(): T
  }

  class StringIterator(s: String) extends AbsIterator {
    type T = Char
    private var i: Int = 0

    def hasNext: Boolean = i < s.length

    def next(): Char = {
      val ch: Char = s charAt i
      i += 1
      ch
    }
  }

  trait RichIterator extends AbsIterator {
    def foreach(f: T => Unit): Unit = while (hasNext) f(next())
  }

  class RichStringIter extends StringIterator("Scala") with RichIterator

  val richStringIter: RichStringIter = new RichStringIter
  richStringIter.foreach(println)

  def test(x: Int): Int = x + 1

  val t1: Int => Int = (x: Int) => test(x)
  (1 to 10).map(t1)


  import scala.util.Random

  object CustomerID {

    def apply(name: String) = s"$name--${Random.nextLong}"

    def unapply(customerID: String): Option[String] = {
      val stringArray: Array[String] = customerID.split("--")
      if (stringArray.tail.nonEmpty) Some(stringArray.head) else None
    }
  }

  Option(1)

  case class User(name: String, age: Int)

  val userBase: immutable.Seq[User] = List(User("Travis", 28),
    User("Kelly", 33),
    User("Jennifer", 44),
    User("Dennis", 23))

  val twentySomethings: immutable.Seq[String] = for (user <- userBase) yield user.name
  // i.e. add this to a list

  twentySomethings.foreach(name => println(name)) // prints Travis Dennis

  class Stack[A] {
    private var elements: List[A] = Nil

    def push(x: A) {
      elements = x :: elements
    }

    def peek: A = elements.head

    def pop(): A = {
      val currentTop = peek
      elements = elements.tail
      currentTop
    }
  }

  val stack: Stack[Int] = new Stack[Int]
  stack.push(1)
  stack.push(2)
  println(stack.pop()) // prints 2
  println(stack.pop())


  class Person private(name: String)

  object Person {
    def apply(name: String) = new Person(name)
  }

  Person("ii")
  Unit

}
