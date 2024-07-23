import * as tf from '@tensorflow/tfjs';
import { ppnetConfig } from './config.js';
import { DEFAULT_CONFIG } from './config.js';
import { protoPartLoss, getProtoClassIdx } from './utils.js';

function resolveConfig (config: ppnetConfig): Required<ppnetConfig> {
    return {
      ...DEFAULT_CONFIG,
      ...config
    }
  };

interface DataPoint extends tf.TensorContainerObject {
  xs: tf.Tensor3D,
  ys: tf.Tensor1D,
};

 /**
   * This function implements custom balanced accuracy metric
   *
   * @param labels - tensor of GT labels
   * @param predictions - tensor of model predictions
   * @returns balanced accuracy, sensitivity, specificity as an array
   */
function balancedAccuracy(labels: tf.Tensor, predictions: tf.Tensor): [tf.Tensor, tf.Tensor, tf.Tensor] {
    return tf.tidy(() => {
    // Calculate true positives, true negatives, false positives, and false negatives
    const yTrue = labels.argMax(1);
    const yPred = predictions.argMax(1);
    const truePositives = yTrue.mul(yPred).sum().cast('float32');
    const trueNegatives = yTrue.equal(0).logicalAnd(yPred.equal(0)).sum().cast('float32');
    const falsePositives = yTrue.equal(0).logicalAnd(yPred.equal(1)).sum().cast('float32');
    const falseNegatives = yTrue.equal(1).logicalAnd(yPred.equal(0)).sum().cast('float32');

    // Calculate sensitivity and specificity
    const epsilon = 1e-7;  // A small number to prevent division by zero
    const sensitivity = truePositives.div(truePositives.add(falseNegatives).add(epsilon)).cast('float32');
    const specificity = trueNegatives.div(trueNegatives.add(falsePositives).add(epsilon)).cast('float32');

    // Calculate balanced accuracy
    const balancedAcc = sensitivity.add(specificity).div(tf.scalar(2));

    return [balancedAcc, sensitivity, specificity];
})};

 /**
   * This function implements the evaluation of model performance
   *
   * @param model - model to evaluate
   * @param dataset - validation dataset
   * @param cfg - model config
   * @returns validation loss, balanced accuracy, sensitivity, specificity
   */
export default async function evaluate (
    model: tf.LayersModel,
    dataset: tf.data.Dataset<DataPoint>,
    cfg: ppnetConfig
): Promise<Record<'val_loss' | 'balanced_acc' | 'sensitivity' | 'specificity', number>> {
    const config = resolveConfig(cfg);

    const protoClassId = getProtoClassIdx(config);
    const ppLoss = protoPartLoss(config, protoClassId);

    let totalLoss = 0;
    let totalBalancedAcc = 0;
    let totalBatches = 0;
    let totalSens = 0;
    let totalSpec = 0;

    await dataset.forEachAsync(async (batch) => {
        const xs = batch.xs;
        const ys = batch.ys;
        const ysClone = ys.clone();

        const output = model.predict(xs) as tf.Tensor[];
        const lossTensor = ppLoss(ys, output[0]);
    

        const lossData = await lossTensor.data() as Float32Array;
        totalLoss += lossData.reduce((acc: number, val: number) => acc + val, 0);
        //console.log('Total loss:', totalLoss);

        const predictions = output[0].slice([0, 0], [-1, config.numClasses]);
        //console.log('Prediction:', predictions);
        const [balancedAccTensor, sensitivityTensor, specificityTensor] = balancedAccuracy(ysClone, predictions);
        //console.log('Acc:', balancedAccTensor);
        const balancedAcc = balancedAccTensor.arraySync() as number;
        //console.log('Balanced acc:', balancedAcc);
        const sensitivity = sensitivityTensor.arraySync() as number;
        const specificity = specificityTensor.arraySync() as number;
        totalBalancedAcc += balancedAcc;
        totalSens += sensitivity;
        totalSpec += specificity;

        totalBatches++;
        ysClone.dispose();
    
    });

    const avgLoss = totalLoss / totalBatches;
    const avgBalancedAcc = totalBalancedAcc / totalBatches;
    const avgSens = totalSens / totalBatches;
    const avgSpec = totalSpec / totalBatches;
    console.log('Average balanced acc: ', avgBalancedAcc)
    console.log('Average sensitivity: ', avgSens)
    console.log('Average specificity: ', avgSpec) 

    return {val_loss: avgLoss, 
        balanced_acc: avgBalancedAcc, 
        sensitivity: avgSens,
        specificity: avgSpec};
};