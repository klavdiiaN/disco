import * as msgpack from 'msgpack-lite'
import * as tf from '@tensorflow/tfjs'

import { makeID } from '../authentication'
import { Client } from '../client'
import { serializeWeights } from '../serialization'
import { Task } from '@/core/task/task'
import { TrainingInformant } from '@/core/training/training_informant'
import * as api from './federated_api'
import { Weights } from '@/types'
/**
 * Class that deals with communication with the centralized server when training
 * a specific task.
 */
export class FederatedClient extends Client {
  clientID: string;
  peer: any;
  modelUpdateIsBasedOnRoundNumber: number;

  /**
   * Prepares connection to a centralized server for training a given task.
   * @param {String} serverURL The URL of the centralized server.
   * @param {Task} task The associated task object.
   * @param {Number} round The training round.
   */
  constructor (serverURL: string, task: Task) {
    super(serverURL, task)
    this.clientID = ''
    this.modelUpdateIsBasedOnRoundNumber = -1 // The server starts at round 0, in the beginning we are behind
  }

  /**
   * Initialize the connection to the server. TODO: In the case of FeAI,
   * should return the current server-side round for the task.
   */
  async connect (): Promise<void> {
    /**
     * Create an ID used to connect to the server.
     * The client is now considered as connected and further
     * API requests may be made.
     */
    this.clientID = makeID(10)
    const response = await api.connect(this.task.taskID, this.clientID)
    if (response.status !== 200) {
      throw new Error('connect api')
    }
  }

  /**
   * Disconnection process when user quits the task.
   */
  async disconnect () {
    const response = await api.disconnect(this.task.taskID, this.clientID)
    return response.status === 200
  }

  private async postWeightsToServer (weights: Weights) {
    const serializedWeights = (await serializeWeights(weights))
    const response = await api.postWeights(
      this.task.taskID,
      this.clientID,
      serializedWeights,
      this.modelUpdateIsBasedOnRoundNumber
    )
    return response.status === 200
  }

  async postMetadata (metadataID, metadata) {
    const response = api.postMetadata(
      this.task.taskID,
      this.modelUpdateIsBasedOnRoundNumber,
      this.clientID,
      metadataID,
      metadata
    )
    return (await response).status === 200
  }

  async getMetadataMap (metadataID) {
    const response = await api.getMetadataMap(
      this.task.taskID,
      this.modelUpdateIsBasedOnRoundNumber,
      this.clientID,
      metadataID
    )
    if (response.status === 200) {
      const body = await response.data
      return new Map(msgpack.decode(body[metadataID]))
    } else {
      return new Map()
    }
  }

  private async getLatestServerRound (): Promise<number> {
    const response = await api.getRound(this.task.taskID, this.clientID)

    if (response.status === 200) {
      return response.data.round
    }
    console.log('Error getting weights: code', response.status)
    return -1
  }

  private async updateLocalModel (model: tf.LayersModel) {
    // get latest model from the server
    const latestModel = await this.getLatestModel()

    // update the model weights
    model.setWeights(latestModel.getWeights())

    console.log('Updated local model')
  }

  private async fetchChangesAndUpdateModel (model: tf.LayersModel) {
    // get server round of latest model
    const serverRound = await this.getLatestServerRound()

    const localRoundIsOld = this.modelUpdateIsBasedOnRoundNumber < serverRound
    if (localRoundIsOld) {
      // update local round
      // TODO need to check that update method did not fail!
      this.modelUpdateIsBasedOnRoundNumber = serverRound
      // update local model from server
      await this.updateLocalModel(model)
    }
  }

  private async fetchServerStatisticsAndUpdateInformant (trainingInformant: TrainingInformant) {
    const serverStatistics = await api.getAsyncWeightInformantStatistics(this.task.taskID, this.clientID)
    trainingInformant.updateWithServerStatistics(serverStatistics.data.statistics)
  }

  async onRoundEndCommunication (model: tf.LayersModel, _: number, trainingInformant: TrainingInformant) {
    await this.postWeightsToServer(model.weights.map((w) => w.read()))
    await this.fetchChangesAndUpdateModel(model)
    await this.fetchServerStatisticsAndUpdateInformant(trainingInformant)
  }

  async onTrainEndCommunication (_: tf.LayersModel, trainingInformant: TrainingInformant) {
    trainingInformant.addMessage('Training finished.')
  }
}
