package akka.actors

import akka.actor.{Actor, Status}
import akka.event.Logging

class PingPongActor extends Actor {

  val log = Logging(context.system, this)

  override def receive: Receive = {
    case "Ping" =>
      log.info("asking ping, telling pong")
      sender() ! "Pong"
    case e =>
      log.info(s"asking $e, telling with exception")
      sender() ! Status.Failure(new Exception(s"unknown message - $e"))
  }
}
