package akka.actors

import akka.actor.{ActorSelection, ActorSystem}
import akka.messages.Requests._
import akka.pattern.ask
import akka.util.Timeout

import scala.concurrent.Future
import scala.concurrent.duration._

/*
* remote akka-in-memory-db-client that asks on remoteAddress of AkkaInMemDb. AkkaInMemDb tells asynchronously.
*/

class AkkaInMemDbClient(remoteAddress: String) {

  private implicit val timeout: Timeout = Timeout(2 seconds)
  private implicit val system: ActorSystem = ActorSystem("LocalSystem")
  private val akkaRemoteDb: ActorSelection = system.actorSelection(s"akka.tcp://AkkaInMemDb@$remoteAddress/user/AkkaInMemDb")

  def set(key: String, value: Object): Future[Any] = {
    akkaRemoteDb ? SetRequest(key, value)
  }

  def get(key: String): Future[Any] = {
    akkaRemoteDb ? GetRequest(key)
  }

}
