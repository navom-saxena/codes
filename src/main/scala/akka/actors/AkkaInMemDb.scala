package akka.actors

import akka.actor.{Actor, ActorRef, ActorSystem, Props, Status}
import akka.event.Logging
import akka.messages.Requests._

import scala.collection.mutable

/* akka in memory db that is started by Main singleton and uses application conf to start on mentioned ip
*  started by plugin application in build.gradle and MainClassName. AkkaInMemDbClientTest creates AkkaInMemDbClient client
*  and asks data. AkkaInMemDb tells asynchronously.
*/

class AkkaInMemDb extends Actor {

  val map: mutable.Map[String, Object] = new mutable.HashMap[String, Object]()
  val log = Logging(context.system, this)

  override def receive: Receive = {
    case SetRequest(key, value) =>
      log.info("received SetRequest - key: {} value: {} . Setting key value", key, value)
      map.put(key, value)
      sender() ! Status.Success
    case GetRequest(key) =>
      log.info("received GetRequest - key: {} . Searching key ...")
      val valueOption: Option[Object] = map.get(key)
      valueOption match {
        case Some(value) => sender() ! value
        case None => sender() ! Status.Failure(KeyNotFoundException(key))
      }
    case o =>
      log.info("received unknown message object: {}", o)
      sender() ! Status.Failure(new ClassNotFoundException(o.toString))
  }

}

object Main extends App {

  implicit val system: ActorSystem = ActorSystem("AkkaInMemDb")
  val actorRef: ActorRef = system.actorOf(Props[AkkaInMemDb], "AkkaInMemDb")

}
