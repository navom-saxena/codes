package akka.cluster

import akka.actor.{Actor, ActorRef, ActorSystem, Props}
import akka.cluster.ClusterEvent.{MemberEvent, UnreachableMember}
import akka.event.Logging

class ClusterController extends Actor {
  val log = Logging(context.system, this)
  val cluster: Cluster = Cluster(context.system)

  override def preStart() {
    cluster.subscribe(self, classOf[MemberEvent], classOf[UnreachableMember])
  }

  override def postStop() {
    cluster.unsubscribe(self)
  }

  override def receive: PartialFunction[Any, Unit] = {
    case x: MemberEvent => log.info("MemberEvent: {}", x)
    case x: UnreachableMember => log.info("UnreachableMember {}: ", x)
  }
}

object Main extends App {
  val system: ActorSystem = ActorSystem("AkkaInMemoryDb")
  val clusterController: ActorRef = system.actorOf(Props[ClusterController], "clusterController")
}