package scalalearning

import scala.collection.immutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.util.{Failure, Success}

// testing map and callbacks, await and Durations

object FuturesFun extends App {

  val errorFuture: Future[Nothing] = Future {
    throw new Exception("Mission Failed")
  }

  //  callback

  add(2, 3).onComplete(result => println(result))

  add(2, 3).onComplete {
    case Success(n) => println(n)
    case Failure(e) => println(e.getMessage)
  }

  add(2, 3).onSuccess { case n => println(n) }
  val listOfFutures: immutable.Seq[Future[Int]] = List(add(1, 2), add(3, 4), add(5, 6))
  val f: Future[Int] = Future(0)
  val g: Future[Int] = Future(1)

  for (sum1 <- add(1, 2);
       sum2 <- add(3, 4);
       sum3 <- add(5, 6)) yield {
    println((sum1, sum2, sum3))
  }

  add(1, 2).flatMap(sum1 =>
    add(3, 4).flatMap(sum2 =>
      add(5, 6).map(sum3 => {
        println((sum1, sum2, sum3))
      })))
  val f1: Future[Unit] = f.map { x =>
    println(x + 100)
    Thread.sleep(5000)
    println("The program waited patiently for this map to finish.")
  }
  Future.sequence(listOfFutures).map(println(_))
  Future.sequence(listOfFutures).map(list =>
    println(list.sum))

  (errorFuture recoverWith {
    case _ => add(1, 2)
  }).map(result => println(result))

  (errorFuture fallbackTo add(1, 2)).map(
    result => println(result))
  val f2: Future[Int] = f andThen {
    case Success(value) =>
      println(100 * value)
      Thread.sleep(10000)
      println("The program waited patiently for this callback to finish.")
    case Failure(exception) => exception.printStackTrace()
  }
  var i, j, k = 0

  def addThenDouble(x: Int, y: Int): Future[Future[Int]] = add(x, y).map(sum => double(sum))

  def add(x: Int, y: Int) = Future {
    x + y
  }

  Thread.sleep(10000)
  Await.ready(f1 zip f2, Duration.Inf)

  /*
  test to show how non-atomic shared resource causes discrepancy and race condition
  1st condition updates shared variable using multiple threads race condition
  2nd condition uses synchronised block, ie locking on shared variable
  3rd condition is single thread update
   */

  def double(n: Int): Future[Int] = Future {
    n * 2
  }

  (1 to 100000).foreach(_ => Future(i = i + 1))
  (1 to 100000).foreach(_ => Future(synchronized(k = k + 1)))
  (1 to 100000).foreach(_ => j = j + 1)
  Thread.sleep(1000)
  println(s"$i $j $k")


}
