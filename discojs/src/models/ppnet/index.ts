// This code is adapted from the GPT model implementation in DISCO //

import * as tf from '@tensorflow/tfjs';
import { Model } from '../model.js';
import { PPNetModel } from './model.js';
import { WeightsContainer} from '../../index.js';
import type { Dataset } from '../../dataset/index.js';
import type { EpochLogs, Prediction, Sample } from '../model.js';
import type { ppnetConfig } from './config.js'; 

export class PPNet extends Model {
    private model!: PPNetModel;

    private constructor() {
        super();
    }

    static async createInstance(numClasses: number | undefined=2): Promise<PPNet> {
        const ppNet = new PPNet();
        const numProts = 10*numClasses;
        const config: ppnetConfig = {
            lr: 0.001,
            imgSize: 224,
            prototypeShape: [numProts, 1, 1, 128],
            featureShape: [7, 7, 2048], // shape of the final convolutional output of the MobileNetV2 model
            pretrainedPath: 'https://storage.googleapis.com/deai-313515.appspot.com/models/mobileNetV2_35_alpha_2_classes/model.json',
            //pretrainedPath: 'file://../model_cardio/model.json',
            numClasses: numClasses,
            batchSizeTrain: 80,
            batchSizeEval: 127,
            batchSizePush: 75,
            validationSplit: 0.1
        };

        await PPNetModel.createInstance(config).then(model => {
          ppNet.model = model;
        });

        return ppNet;
    }

  /*static setNumClasses(numClasses: number): void {
      PPNet.numClassesDefault = numClasses; // Method to set default numClasses externally
  }*/

  get config (): Required<ppnetConfig> {
    return this.model.getPPNetConfig
  };

  override get weights(): WeightsContainer {
    if (!this.model) {
        console.error("Model is not initialized yet.");
        return new WeightsContainer([]); // or handle this scenario appropriately
    }
    return new WeightsContainer(this.model.weights.map((w) => w.read()));
}

  override set weights(ws: WeightsContainer) {
      if (!this.model) {
          console.error("Model is not initialized yet.");
          return; // or schedule the setting for when the model is ready
      }
      this.model.setWeights(ws.weights);
  }
  
    static async deserialize(data: PPNetSerialization, numClasses?: number): Promise<PPNet> {
      const ppNet = await PPNet.createInstance(numClasses);
      ppNet.weights = data.weights;
      return ppNet;
  }  
  
  serialize (): PPNetSerialization {
    return {
      weights: this.weights,
      config: this.config
    }
  }

    /**
   * The PPNet train method wraps the model.fitDataset call in a for loop to act as a generator (of logs)
   * This allows for getting logs and stopping training without callbacks.
   *
   * @param trainingData training dataset
   * @param validationData validation dataset (optional)
   * @param epochs the number of passes of the training dataset
   * @param clientNumber the number of the currently active client
   * @param pushData dataset to project prototypes (optiobal)
   */
  override async *train(
    trainingData: Dataset,
    validationData?: Dataset,
    epochs = 1,
    clientNumber?: number,
    pushData?: Dataset
  ): AsyncGenerator<EpochLogs, void> {
    let logs: tf.Logs | undefined;

    const trainingArgs: tf.ModelFitDatasetArgs<tf.TensorContainer> = {
      epochs: 1, // force fitDataset to do only one epoch because it is wrapped in a for loop
      validationData,
      callbacks: {
        onEpochEnd: (_, cur) => {
          logs = cur;
          if (logs !== undefined && cur !== undefined) {
            logs.loss = cur.val_loss;
          }
        },
      },
    };
    for (let epoch = 0; epoch < epochs; epoch++) {
      trainingArgs.epochs = epoch;
      await this.model.fitDataset(trainingData, trainingArgs, clientNumber, pushData);

      if (logs === undefined) {
        throw new Error("epoch didn't gave any logs");
      }
      const { loss, val_loss, balanced_acc, peakMemory } = logs;
      if (
        loss === undefined || isNaN(loss)
      ) {
        throw new Error("epoch gave invalid logs");
      }

      const structuredLogs: EpochLogs = {
        epoch,
        peakMemory,
        training: {
          loss: logs.loss
        }
    };

    if (validationData !== undefined) {
      if(val_loss === undefined || isNaN(val_loss) ||
        balanced_acc === undefined || isNaN(balanced_acc)) {
        throw new Error("Invalid validation logs");
      }
      structuredLogs.validation = { accuracy: logs.balanced_acc, loss: logs.val_loss}
    }
    yield structuredLogs
  }
}

  override predict (input: Sample): Promise<Prediction> {
    const ret = this.model.predict(input)
    if (Array.isArray(ret)) {
      throw new Error('prediction yield many Tensors but should have only returned one')
    }

    return Promise.resolve(ret)
  }

  [Symbol.dispose](): void{
    console.log("Disposing model")
    if (this.model.optimizer !== undefined) {
      this.model.optimizer.dispose()
    }
    // Some tensors are not cleaned up when model.dispose is called 
    // So we dispose them manually
    //this.model.disposeRefs()
    //this.model.dispose()
  }
}

export type PPNetSerialization = {
  weights: WeightsContainer
  config?: ppnetConfig
}