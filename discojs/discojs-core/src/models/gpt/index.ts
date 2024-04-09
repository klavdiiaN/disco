/**
 * this code is taken from gpt-tfjs with modifications from @peacefulotter and @lukemovement
 **/

import * as tf from '@tensorflow/tfjs'
import { PreTrainedTokenizer } from '@xenova/transformers';

import { WeightsContainer } from '../../index.js'
import type { Dataset } from '../../dataset/index.js'
import { Sink } from '../../utils/event_emitter.js'

import { Model } from '../model.js'
import { GPTForCausalLM } from './model.js'
import type { EpochLogs, Prediction, Sample } from '../model.js'
import type { GPTConfig } from './config.js'


export class GPT extends Model {
  private readonly model: GPTForCausalLM

  constructor (partialConfig?: GPTConfig) {
    super()
    this.model = new GPTForCausalLM(partialConfig)
  }

  override get weights (): WeightsContainer {
    return new WeightsContainer(this.model.weights.map((w) => w.read()))
  }

  override set weights (ws: WeightsContainer) {
    this.model.setWeights(ws.weights)
  }

  /**
   * The GPT train methods wraps the model.fitDataset call in a for loop to act as a generator (of logs)
   * This allows for getting logs and stopping training without callbacks.
   *
   * @param trainingData training dataset
   * @param validationData validation dataset
   * @param epochs the number of passes of the training dataset
   * @param tracker
   */
  override async * train (
    trainingData: Dataset,
    validationData?: Dataset,
    epochs = 1,
    tracker = new Sink()
  ): AsyncGenerator<EpochLogs, void> {
    this.model.compile() // init the optimizer
    let logs: tf.Logs | undefined
    const trainingArgs: tf.ModelFitDatasetArgs<tf.TensorContainer> = {
      epochs: 1, // force fitDataset to do only one epoch because it is wrapped in a for loop
      validationData,
      callbacks: {
        onEpochEnd: (_, cur) => {
          logs = cur
          if (logs !== undefined && cur !== undefined) {
            logs.loss = cur.val_loss
          }
        },
        onBatchBegin: () => { tracker.emit('batchBegin', undefined) },
        onBatchEnd: () => { tracker.emit('batchEnd', undefined) }
      }
    }
    for (let i = 0; i < epochs; i++) {
      await this.model.fitDataset(trainingData, trainingArgs)
      yield logs
      if (logs !== undefined && logs.training_loss < 0.01) {
        console.log("Early stopping")
        break
      }
    }
  }

  override predict (input: Sample): Promise<Prediction> {
    const ret = this.model.predict(input)
    if (Array.isArray(ret)) {
      throw new Error('prediction yield many Tensors but should have only returned one')
    }

    return Promise.resolve(ret)
  }

  async generate (input: string, tokenizer: PreTrainedTokenizer, newTokens: number = 10): Promise<string> {
    const { input_ids: tokens } = await tokenizer(input, { return_tensor: false}) as { input_ids: number[] }

    const generationConfig = {
      maxNewTokens: newTokens,
      temperature: 1.0,
      doSample: false,
      topK: null
    }
    const predictedTokens = await this.model.generate(tokens, generationConfig)
    const generatedWords = tokenizer.decode(predictedTokens[0])
    return generatedWords
  }

  static deserialize (weights: WeightsContainer): Model {
    const model = new GPT()
    model.weights = weights
    return model
  }

  serialize (): WeightsContainer {
    return this.weights
  }
}
