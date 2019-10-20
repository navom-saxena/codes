package akka

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.actors.PingPongActor
import akka.pattern.ask
import akka.util.Timeout
import org.scalatest.{FunSpecLike, Matchers}

import scala.concurrent.duration._
import scala.concurrent.{Await, Future}

class PingPongActorTest extends FunSpecLike with Matchers {

  implicit val system: ActorSystem = ActorSystem()
  implicit val timeout: Timeout = Timeout(1 seconds)
  val pingPongActor: ActorRef = system.actorOf(Props(classOf[PingPongActor]))

  describe("Ping Pong Actor") {
    it("should respond with Pong") {

      val responseFuture: Future[Any] = pingPongActor ? "Ping"
      val response: String = Await.result(responseFuture.mapTo[String], 1 seconds)
      assert(response == "Pong")

    }

    it("should fail n unknown message") {

      val responseFuture: Future[Any] = pingPongActor ? "unknown message"
      intercept[Exception] {
        Await.result(responseFuture.mapTo[String], 1 seconds)
      }

    }
  }

}
