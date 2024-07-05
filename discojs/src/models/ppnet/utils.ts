import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import type { ppnetConfig } from './config.js';
import { DEFAULT_CONFIG } from './config.js';

function resolveConfig (config: ppnetConfig): Required<ppnetConfig> {
    return {
      ...DEFAULT_CONFIG,
      ...config
    }
  };

export function getProtoClassIdx (cfg: any): tf.Tensor {
    //const config = resolveConfig(cfg);
    const config = Object.assign({}, cfg);
    const numClasses = config.numClasses;
    const numPrototypes = config.prototypeShape[0];

    const numPrototypePerClasses = Math.floor(numPrototypes / numClasses);
    const protoClassId = tf.zeros([numPrototypes, numClasses]);
    let protoClassIdBuffer = tf.buffer(protoClassId.shape, protoClassId.dtype, protoClassId.dataSync())

    for (let j = 0; j < numPrototypes; j++) {
        protoClassIdBuffer.set(1, j, Math.floor(j / numPrototypePerClasses));
    }

    return protoClassIdBuffer.toTensor();
};

export function protoPartLoss (cfg: any, protoClassId: tf.Tensor) {
    return (yTrue: tf.Tensor, yPred: tf.Tensor): tf.Tensor => {
        return tf.tidy(() => {
        const predictions = yPred.slice([0, 0], [-1, cfg.numClasses]);
        //console.log('Predictions:', predictions.shape);
        const minDistances = yPred.slice([0, cfg.numClasses], [-1, cfg.prototypeShape[0]]);
        //console.log('Min distances:', minDistances);
        //const yTrueInt32 = tf.cast(yTrue, 'int32');
        //const oneHotLabels = tf.oneHot(yTrueInt32, cfg.numClasses);
        //console.log('Labels:', yTrue.shape);
        //console.log('OneHot Labels:', oneHotLabels.shape);
        const crossEntropy = tf.losses.softmaxCrossEntropy(yTrue, predictions);
        //console.log('CrossEntropy:', crossEntropy);
        const labels = yTrue.argMax(1);
        //console.log('Labels:', labels);

        // cluster cost
        const maxDistance = cfg.prototypeShape[1] * cfg.prototypeShape[2] * cfg.prototypeShape[3];
        const prototypesOfCorrectClass = tf.transpose(protoClassId.gather(labels, 1));

        const invertedDistances = tf.max(
            tf.mul(tf.sub(maxDistance, minDistances), prototypesOfCorrectClass),
            1
        );

        const clusterCost = tf.mean(tf.sub(maxDistance, invertedDistances)); 
        //console.log('Cluster Cost:', clusterCost);

        // separation cost
        const prototypesOfWrongClass = tf.sub(tf.scalar(1), prototypesOfCorrectClass);
        const invertedDistancesNontarget = tf.max(
            tf.mul(tf.sub(maxDistance, minDistances), prototypesOfWrongClass),
            1
        );

        const separationCost = tf.mean(tf.sub(maxDistance, invertedDistancesNontarget));
        //console.log('Separation cost:', separationCost);
        const finalLoss = tf.addN([crossEntropy,
            tf.mul(tf.scalar(0.8), clusterCost),
            tf.mul(tf.scalar(-0.08), separationCost)
        ]);

        labels.dispose(); // Dispose tensors that are not needed after their use
            prototypesOfCorrectClass.dispose();
            prototypesOfWrongClass.dispose();
            invertedDistances.dispose();
            invertedDistancesNontarget.dispose();

            return finalLoss;
        });
    }
};

export function readJsonFile(filePath: string) {
    try {
        const rawData = fs.readFileSync(filePath, 'utf8');
        const data = JSON.parse(rawData);
        return data;
    } catch (error) {
        console.error("Error reading or parsing the JSON file:", error);
        return null;
    }
}

