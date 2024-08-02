// This script implements the custom training loop for ProtoPNet in DISCO //
// adapted from https://github.com/EPFLiGHT/MyTH 

import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import type { ppnetConfig } from './config.js';
import { DEFAULT_CONFIG } from './config.js';
import type { TrainingCallbacks } from './types.js';
import { protoPartLoss, getProtoClassIdx } from './utils.js';
import evaluate from './evaluate.js';
import { pushPrototypes } from './pushPrototype.js';
import { L2Convolution_} from './model.js';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';

const mkdir = promisify(fs.mkdir);

function resolveConfig (config: ppnetConfig): Required<ppnetConfig> {
    return {
      ...DEFAULT_CONFIG,
      ...config
    }
  };

/**
 * The train function implements the model training across three stages:
 * - JOINT all network layers except for pretrained feature extractor are trained as usual
 * - PUSH prototypes projection on the nearset latent patches
 * - LAST LAYER OPTIMIZATION only the final layer is trained
 * In the original implementation, there is also a WARM stage in the very beginning which trains only the additional convolutional layers. 
 * After that, during JOINT, the feature extractor weights are unfrozen and trained together with other layers which is not our case. So we don't implement WARM as a separate stage.
 *
 * @param model - ProtoPNet to train
 * @param ds - training dataset
 * @param cfg - model config
 * @param epoch - current epoch number
 * @param callbacks - callback functions for communication
 * @param evalDs - validation dataset
 * @param clientNumber - index number of the currently training client (necessary to save the model and prototypes to the corresponding folder) or a string 'Local' for local training
 * @params pushDs - dataset for push operation (optional)
 */
export async function train (
    model: tf.LayersModel,
    ds: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
    cfg: ppnetConfig,
    epoch: number,
    callbacks: TrainingCallbacks,
    evalDs: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
    clientNumber: number | string | undefined=0,
    pushDS?: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
): Promise<void>{
    const config = resolveConfig(cfg);
    const protoClassId = getProtoClassIdx(config);
    const ppLoss = protoPartLoss(config,protoClassId); // custom loss function
    console.log(`Using backend: ${tf.getBackend()}`);
    let logs: tf.Logs | undefined
    let loss;

    // freeze pretrained convolutional layers
    const freeze = model.getLayer('features');
    freeze.trainable = false;

    console.log('JOINT'); // indication that all layers are being trained
    const opt = tf.train.adam(config.lr);

    //console.log(tf.memory());
    await ds.forEachAsync(async (batch) => {
        const xs = batch.xs;
        const ys = batch.ys;

        const lossFn: () => tf.Scalar = () => {
            return tf.tidy(() => {
                const output = model.predict(xs) as tf.Tensor[];
                return ppLoss(ys, output[0]) as tf.Scalar;
            });
        };
        
        const lossTensor = opt.minimize(lossFn, true); // gradients are zeroed automatically
        //console.log('Loss Tensor:', lossTensor);
        if (lossTensor) {
            loss = await lossTensor.array();  // Ensure to await the promise
            lossTensor.dispose();
        };
        xs.dispose();
        ys.dispose()
    });

    console.log('Loss calculation finished');        
    logs = await evaluate(model, evalDs, config);
    
    console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
    opt.dispose();
    
    // we perform push operation each 10 epochs
    if (epoch > 0 && epoch%10 === 0){
        const modelEpochDir = `../models-client${clientNumber}`
        await mkdir(modelEpochDir, { recursive: true });
        
        let fileName = path.join(modelEpochDir, `epoch${epoch}-beforePush`)
        await model.save(`file://${fileName}`);
        console.log('Before push model saved')

        const opt = tf.train.adam(config.lr);
        console.log('PUSH');
        try {
            if (pushDS){
                await pushPrototypes(config, pushDS, model, null, 1, '../', epoch, 'prot', 'self_act', 'prot_bb', true, true, clientNumber);
            } else {
            await pushPrototypes(config, ds, model, null, 1, '../', epoch, 'prot', 'self_act', 'prot_bb', true, true, clientNumber);
            }
        } catch (error) {
            console.error('Error in push:', error)  
        }
        // after push we need to optimize the weights of the final linear layer while keeping all other parameters frozen
        const addOnLayersToFreeze1 = model.getLayer('additional_conv_1');
        const addOnLayersToFreeze2 = model.getLayer('additional_conv_2');
        const protsLayer = model.getLayer('l2_convolution');
        if (protsLayer instanceof L2Convolution_) {
            protsLayer.freezePrototypes();
        };

        addOnLayersToFreeze1.trainable, addOnLayersToFreeze2.trainable = false, false;
        // final layer optimization is happening //
        console.log('LAST LAYER OPTIMIZATION');
        // optimize for some number of iterations
        for(let i=0; i<8; i++){
            console.log(tf.memory());
            await ds.forEachAsync(async (batch) => {
                const xs = batch.xs;
                const ys = batch.ys;
    
                const lossFn: () => tf.Scalar = () => {
                    return tf.tidy (() => {
                    const output = model.predict(xs) as tf.Tensor[];
                    return ppLoss (ys, output[0]) as tf.Scalar;
                })
                };
                
                const lossTensor = opt.minimize(lossFn, true); // gradients are zeroed automatically
                if (lossTensor) {
                    loss = await lossTensor.array();
                    lossTensor.dispose();
                };
                xs.dispose();
                ys.dispose()
            });
            console.log('iteration: ', i, '--- train loss: ', loss);
        };

        logs = await evaluate(model, evalDs, config);
        
        // unfreeze the layers
        addOnLayersToFreeze1.trainable, addOnLayersToFreeze2.trainable = true, true;
        if (protsLayer instanceof L2Convolution_) {
            protsLayer.unfreezePrototypes();
        };
        console.log('LAST LAYER OPTIMIZATION COMPLETE');
        console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
        opt.dispose();

        fileName = path.join(modelEpochDir, `epoch${epoch}-final`)
        await model.save(`file://${fileName}`);
        console.log('Final model saved')
        };
        await callbacks.onEpochEnd?.(epoch, logs)
    }