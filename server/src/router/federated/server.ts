import express from 'express'
import WebSocket from 'ws'
import { v4 as randomUUID } from 'uuid'

import { List, Map } from 'immutable'
import msgpack from 'msgpack-lite'

import {
  client,
  tf,
  serialization,
  AsyncInformant,
  Task,
  TaskID,
  aggregator as aggregators,
  WeightsContainer,
  MetadataKey,
  MetadataValue
} from '@epfml/discojs-node'

import { Server } from '../server'

import messages = client.federated.messages
import AssignNodeID = client.messages.AssignNodeID

import MessageTypes = client.messages.type

/**
 * Represents a log entry for a given request. Consists of:
 * - the request type corresponding to the exchanged message
 * - the node id who made the request
 * - the task id for which the request was made
 * - the round for which the request was made
 * - the timestamp at which the request was made
 */
interface Log {
  timestamp: Date
  task: TaskID
  round: number
  nodeId: client.NodeID
  type: MessageTypes
}

export class Federated extends Server {
  /**
   * Aggregators for each hosted task.
   */
  private aggregators = Map<TaskID, aggregators.Aggregator>()
  /**
   * Promises containing the current round's results. To be awaited on when providing clients
   * with the most recent result.
   */
  private results = Map<TaskID, Promise<WeightsContainer>>()
  /**
   * Training informants for each hosted task.
   */
  private informants = Map<TaskID, AsyncInformant<WeightsContainer>>()
  /**
   * Contains metadata used for training by clients for a given task and round.
   * Stored by task id, round number, node id and metadata key.
  */
  private metadataMap = Map<TaskID, Map<number, Map<client.NodeID, Map<MetadataKey, MetadataValue>>>>()
  // TODO use real log system
  /**
  * Logs of successful requests made to the server.
  */
  private logs = List<Log>()

  private rounds = Map<TaskID, number>()

  protected get description (): string {
    return 'Disco Federated Server'
  }

  protected buildRoute (task: Task): string {
    return `/${task.taskID}`
  }

  public isValidUrl (url: string | undefined): boolean {
    const splittedUrl = url?.split('/')

    return (
      splittedUrl !== undefined &&
      splittedUrl.length === 3 &&
      splittedUrl[0] === '' &&
      this.isValidTask(splittedUrl[1]) &&
      this.isValidWebSocket(splittedUrl[2])
    )
  }

  /**
   * Loop creating an aggregation result promise at each round.
   * Because clients contribute to the round asynchronously, a promise is used to let them wait
   * until the server has aggregated the weights. This loop creates a promise whenever the previous
   * one resolved and awaits until it resolves. The promise is used in createPromiseForWeights.
   * @param aggregator The aggregation handler
   */
  private async storeAggregationResult (aggregator: aggregators.Aggregator): Promise<void> {
    // Create a promise on the future aggregated weights
    const result = aggregator.receiveResult()
    // Store the promise such that it is accessible from other methods
    this.results = this.results.set(aggregator.task.taskID, result)
    // The promise resolves once the server received enough contributions (through the handle method)
    // and the aggregator aggregated the weights.
    await result
    // Update the server round with the aggregator round
    this.rounds = this.rounds.set(aggregator.task.taskID, aggregator.round)
    // Create a new promise for the next round
    void this.storeAggregationResult(aggregator)
  }

  protected initTask (task: Task, model: tf.LayersModel): void {
    const aggregator = new aggregators.MeanAggregator(task, model)

    this.aggregators = this.aggregators.set(task.taskID, aggregator)
    this.informants = this.informants.set(task.taskID, new AsyncInformant(aggregator))
    this.rounds = this.rounds.set(task.taskID, 0)

    void this.storeAggregationResult(aggregator)
  }

  /**
   * This method is called when a client sends its contribution to the server. The server
   * first adds the contribution to the aggregator and then replies with the aggregated weights
   *
   * @param msg the client message received of type SendPayload which contains the local client's weights
   * @param task the task for which the client is contributing
   * @param clientId the clientID of the contribution
   * @param ws the websocket through which send the aggregated weights
   */
  private async addContributionAndSendModel (msg: messages.SendPayload, task: Task,
    clientId: client.NodeID, ws: WebSocket): Promise<void> {
    const { payload, round } = msg
    const aggregator = this.aggregators.get(task.taskID)

    if (!(Array.isArray(payload) &&
      payload.every((e) => typeof e === 'number'))) {
      throw new Error('received invalid weights format')
    }
    if (aggregator === undefined) {
      throw new Error(`received weights for unknown task: ${task.taskID}`)
    }

    // It is important to create a promise for the weights BEFORE adding the contribution
    // Otherwise the server might go to the next round before sending the
    // aggregated weights. Once the server has aggregated the weights it will
    // send the new weights to the client.
    // Use the void keyword to explicity avoid waiting for the promise to resolve
    this.createPromiseForWeights(task, aggregator, ws)
      .catch(console.error)

    const serialized = serialization.weights.decode(payload)
    // Add the contribution to the aggregator,
    // which returns False if the contribution is too old
    if (!aggregator.add(clientId, serialized, round, 0)) {
      console.info('Dropped contribution from client', clientId, 'for round', round)
    }
  }

  /**
   * This method is called after received a local update.
   * It puts the client on hold until the server has aggregated the weights
   * by creating a Promise which will resolve once the server has received
   * enough contributions. Relying on a promise is useful since clients may
   * send their contributions at different times and a promise lets the server
   * wait asynchronously for the results
   *
   * @param task the task to which the client is contributing
   * @param aggregator the server aggregator, in order to access the current round
   * @param ws the websocket through which send the aggregated weights
   */
  private async createPromiseForWeights (
    task: Task,
    aggregator: aggregators.Aggregator,
    ws: WebSocket): Promise<void> {
    const promisedResult = this.results.get(task.taskID)
    if (promisedResult === undefined) {
      throw new Error(`result promise was not set for task ${task.taskID}`)
    }

    // Wait for aggregation result to resolve with timeout, giving the network a time window
    // to contribute to the model
    void Promise.race([promisedResult, client.utils.timeout()])
      .then((result) =>
      // Reply with round - 1 because the round number should match the round at which the client sent its weights
      // After the server aggregated the weights it also incremented the round so the server replies with round - 1
        [result, aggregator.round - 1] as [WeightsContainer, number])
      .then(async ([result, round]) =>
        [await serialization.weights.encode(result), round] as [serialization.weights.Encoded, number])
      .then(([serialized, round]) => {
        const msg: messages.ReceiveServerPayload = {
          type: MessageTypes.ReceiveServerPayload,
          round,
          payload: serialized
        }
        ws.send(msgpack.encode(msg))
      })
      .catch(console.error)
  }

  protected handle (
    task: Task,
    ws: WebSocket,
    model: tf.LayersModel,
    req: express.Request
  ): void {
    const taskAggregator = this.aggregators.get(task.taskID)
    if (taskAggregator === undefined) {
      throw new Error('connecting to a non-existing task')
    }
    // Client id of the message sender
    let clientId = randomUUID()
    while (!taskAggregator.registerNode(clientId)) {
      clientId = randomUUID()
    }

    ws.on('message', (data: Buffer) => {
      const msg = msgpack.decode(data)

      if (msg.type === MessageTypes.ClientConnected) {
        this.logsAppend(task.taskID, clientId, MessageTypes.ClientConnected, 0)

        let aggregator = this.aggregators.get(task.taskID)
        if (aggregator === undefined) {
          aggregator = new aggregators.MeanAggregator(task)
          this.aggregators = this.aggregators.set(task.taskID, aggregator)
        }
        console.info('client', clientId, 'joined', task.taskID)

        const msg: AssignNodeID = {
          type: MessageTypes.AssignNodeID,
          id: clientId
        }
        ws.send(msgpack.encode(msg))
      } else if (msg.type === MessageTypes.SendPayload) {
        this.logsAppend(task.taskID, clientId, MessageTypes.SendPayload, msg.round)

        if (model === undefined) {
          throw new Error('aggregator model was not set')
        }
        this.addContributionAndSendModel(msg, task, clientId, ws)
          .catch(console.error)
      } else if (msg.type === MessageTypes.ReceiveServerStatistics) {
        const statistics = this.informants
          .get(task.taskID)
          ?.getAllStatistics()

        const msg: messages.ReceiveServerStatistics = {
          type: MessageTypes.ReceiveServerStatistics,
          statistics: statistics ?? {}
        }

        ws.send(msgpack.encode(msg))
      } else if (msg.type === MessageTypes.ReceiveServerPayload) {
        this.logsAppend(task.taskID, clientId, MessageTypes.ReceiveServerPayload, 0)
        const aggregator = this.aggregators.get(task.taskID)
        if (aggregator === undefined) {
          throw new Error(`requesting round of unknown task: ${task.taskID}`)
        }
        if (model === undefined) {
          throw new Error('aggregator model was not set')
        }

        this.createPromiseForWeights(task, aggregator, ws)
          .catch(console.error)
      } else if (msg.type === MessageTypes.SendMetadata) {
        const { round, key, value } = msg

        this.logsAppend(task.taskID, clientId, MessageTypes.SendMetadata, round)

        if (this.metadataMap.hasIn([task.taskID, round, clientId, key])) {
          throw new Error('metadata already set')
        }
        this.metadataMap = this.metadataMap.setIn(
          [task, round, clientId, key],
          value
        )
      } else if (msg.type === MessageTypes.ReceiveServerMetadata) {
        const key = msg.metadataId
        const round = Number.parseInt(msg.round, 0)

        const taskMetadata = this.metadataMap.get(task.taskID)

        if (!Number.isNaN(round) && round >= 0 && taskMetadata !== undefined) {
          // Find the most recent entry round-wise for the given task (upper bounded
          // by the given round). Allows for sporadic entries in the metadata map.
          const latestRound = taskMetadata.keySeq().max() ?? round

          // Fetch the required metadata from the general metadata structure stored
          // server-side and construct the queried metadata's map accordingly. This
          // essentially creates a "ID -> metadata" single-layer map.
          const queriedMetadataMap = Map(
            taskMetadata
              .get(latestRound, Map<string, Map<string, string>>())
              .filter((entries) => entries.has(key))
              .mapEntries(([id, entries]) => [id, entries.get(key)])
          )

          this.logsAppend(task.taskID, clientId, MessageTypes.ReceiveServerMetadata, round)

          const msg: messages.ReceiveServerMetadata = {
            type: MessageTypes.ReceiveServerMetadata,
            taskId: task.taskID,
            nodeId: clientId,
            key,
            round: round,
            metadataMap: Array.from(queriedMetadataMap)
          }

          ws.send(msgpack.encode(msg))
        }
      }
    })
  }

  /**
   * Appends a request to the logs.
   * @param taskId The task id for which the request was made
   * @param nodeId The node id who made the request
   * @param type The request type
   * @param round The round for which the request was made
   */
  private logsAppend (
    taskId: TaskID,
    nodeId: client.NodeID,
    type: MessageTypes,
    round: number | undefined = undefined
  ): void {
    if (round === undefined) {
      return
    }

    this.logs = this.logs.push({
      timestamp: new Date(),
      task: taskId,
      round,
      nodeId,
      type
    })
  }
}
