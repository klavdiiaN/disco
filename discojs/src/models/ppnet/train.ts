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
//import '@tensorflow/tfjs-node-gpu';

//tf.env().set('IS_NODE', true);  // Set if not already set in a Node.js environment
//tf.env().set('DEBUG', true);  // Enable detailed logging

const mkdir = promisify(fs.mkdir);


function resolveConfig (config: ppnetConfig): Required<ppnetConfig> {
    return {
      ...DEFAULT_CONFIG,
      ...config
    }
  };

interface DataPoint extends tf.TensorContainerObject {
    xs: tf.Tensor,
    ys: tf.Tensor,
  };

/*export async function train (
    model: tf.LayersModel,
    ds: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
    cfg: ppnetConfig,
    epochs: number,
    //currentEpoch:number,
    callbacks: TrainingCallbacks,
    evalDs: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>
): Promise<void>{
    const config = resolveConfig(cfg);
    const opt = tf.train.adam(config.lr);
    //const batchedData = ds.batch(config.batchSizeTrain);
    console.log(`Using backend: ${tf.getBackend()}`);
    console.log('WARM');
    console.log('Number of epochs:', epochs);
    for (let epoch=0; epoch<epochs; epoch++){
        let logs: tf.Logs | undefined
        if (epoch === 1){
            const unfreeze = model.getLayer('features');
            unfreeze.trainable = true;
            console.log('JOINT');
        };
        let loss;
        await ds.forEachAsync((batch) => {
            const xs = batch.xs;
            const ys = batch.ys;
            //console.log(xs.shape, ys.shape)

            //console.log(`Batch X shape: ${xs.shape}, Y shape: ${ys.shape}`);
            /*if (xs.shape[0] !== config.batchSizeTrain) {
                console.log('Skip the last batch')
                return;  // Skip processing this batch
            };*/

            /*const lossFn: () => tf.Scalar = () => {
                const output = model.predict(xs) as tf.Tensor[];
                //console.log('output:', output);
                return ppLoss (ys, output[0]) as tf.Scalar;
            };
            
            //console.log('Batch processing finished');
            const lossTensor = opt.minimize(lossFn, true); // gradients are zeroed automatically
            //console.log('Loss Tensor:', lossTensor);
            loss = lossTensor?.array();
            //console.log('Loss:', loss);
            //lossTensor?.dispose()
        });
        console.log('Loss calculation finished');        
        logs = await evaluate(model, evalDs, config);
        
        console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
        if (epoch > 0 && epoch%10 === 0){
            console.log('PUSH');
            await pushPrototypes(config, ds, model, null, 1, '../', 10, 'prot', 'self_act', 'prot_bb', false, true);
            const featuresToFreeze = model.getLayer('features');
            const addOnLayersToFreeze = model.getLayer('addOnLayers');
            const protsToFreeze = model.getLayer('prototypes'); // incorrect, implement !!
    
            featuresToFreeze.trainable, addOnLayersToFreeze.trainable, protsToFreeze.trainable = false, false, false;
            // final layer optimization is happening //
            console.log('LAST LAYER OPTIMIZATION');

            for(let i=0; i<12; i++){
                await ds.forEachAsync((batch) => {
                    const xs = batch.xs;
                    const ys = batch.ys;
        
                    //console.log(`Batch X shape: ${xs.shape}, Y shape: ${ys.shape}`);
                    if (xs.shape[0] !== config.batchSizeTrain) {
                        console.log('Skip the last batch')
                        return;  // Skip processing this batch
                    };
        
                    const lossFn: () => tf.Scalar = () => {
                        const output = model.predict(xs) as tf.Tensor[];
                        return ppLoss(ys, output[0]) as tf.Scalar;
                    };
        
                    const lossTensor = opt.minimize(lossFn); // gradients are zeroed automatically
                    loss = lossTensor?.array();
                    lossTensor?.dispose()
                });
                console.log('iteration: ', i, '--- train loss: ', loss);
            };

            logs = await evaluate(model, evalDs, config);
            
            featuresToFreeze.trainable, addOnLayersToFreeze.trainable, protsToFreeze.trainable = true, true, true;
            console.log('LAST LAYER OPTIMIZED');
            console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
          };

          await callbacks.onEpochEnd?.(epoch, logs)
        };
        opt.dispose();
    };*/


    export async function train (
        model: tf.LayersModel,
        ds: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
        cfg: ppnetConfig,
        epoch: number,
        callbacks: TrainingCallbacks,
        evalDs: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
        clientNumber: number | undefined=0,
        pushDS?: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
    ): Promise<void>{
        const config = resolveConfig(cfg);
        const protoClassId = getProtoClassIdx(config);
        const ppLoss = protoPartLoss(config,protoClassId);
        console.log(`Using backend: ${tf.getBackend()}`);
        // Log all available backend names
        //console.log('Available backends:', Object.keys(tf.engine().registryFactory));
        let logs: tf.Logs | undefined
        let loss;

        if (epoch >= 0 && epoch < 5){            
            const freeze = model.getLayer('features');
            freeze.trainable = false;
            console.log('WARM');
            const opt = tf.train.adam(config.lr);
            console.log(tf.memory());
            await ds.forEachAsync(async (batch) => {
                //console.log(tf.memory());
                //const xs = batch.xs;
                //const ys = batch.ys;
    
                const lossFn: () => tf.Scalar = () => {
                    return tf.tidy (() => {
                    const output = model.predict(batch.xs) as tf.Tensor[];
                    //console.log('output:', output);
                    return ppLoss (batch.ys, output[0]) as tf.Scalar;
                })
                };
                
                //console.log('Batch processing finished');
                const lossTensor = opt.minimize(lossFn, true); // gradients are zeroed automatically
                if (lossTensor) {
                    loss = await lossTensor.array();  // Ensure to await the promise
                    lossTensor.dispose();
                };
                //console.log(tf.memory());
            });

            console.log('Loss calculation finished');        
            logs = await evaluate(model, evalDs, config);
            
            console.log('epoch: ', epoch, '--- train loss: ', loss, '---balanced acc: ', logs.balanced_acc);
            await callbacks.onEpochEnd?.(epoch, logs)
            opt.dispose();
        };

        if (epoch >= 5){
            const unfreeze = model.getLayer('features');
            unfreeze.trainable = false;
            console.log('JOINT');
            const opt = tf.train.adam(config.lr);

            console.log(tf.memory());
            await ds.forEachAsync(async (batch) => {
                const xs = batch.xs;
                const ys = batch.ys;
    
                const lossFn: () => tf.Scalar = () => {
                    return tf.tidy(() => {
                        const output = model.predict(xs) as tf.Tensor[];
                        return ppLoss(ys, output[0]) as tf.Scalar;
                    });
                };
                
                //console.log('Batch processing finished');
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
            
            if (epoch%10 === 0){
                //const timestamp = new Date().getTime(); 
                //const modelEpochDir = `../models-${timestamp}`; //
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
                const featuresToFreeze = model.getLayer('features');
                const addOnLayersToFreeze1 = model.getLayer('additional_conv_1');
                const addOnLayersToFreeze2 = model.getLayer('additional_conv_2');
                const protsLayer = model.getLayer('l2_convolution');
                if (protsLayer instanceof L2Convolution_) {
                    protsLayer.freezePrototypes();
                };
        
                featuresToFreeze.trainable, addOnLayersToFreeze1.trainable, addOnLayersToFreeze2.trainable = false, false, false;
                // final layer optimization is happening //
                console.log('LAST LAYER OPTIMIZATION');
                //let logs: tf.Logs | undefined
                //let loss;
    
                for(let i=0; i<8; i++){
                    console.log(tf.memory());
                    await ds.forEachAsync(async (batch) => {
                        const xs = batch.xs;
                        const ys = batch.ys;
            
                        const lossFn: () => tf.Scalar = () => {
                            return tf.tidy (() => {
                            const output = model.predict(xs) as tf.Tensor[];
                            //console.log('output:', output);
                            return ppLoss (ys, output[0]) as tf.Scalar;
                        })
                        };
                        
                        const lossTensor = opt.minimize(lossFn, true); // gradients are zeroed automatically
                        if (lossTensor) {
                            loss = await lossTensor.array();  // Ensure to await the promise
                            lossTensor.dispose();
                        };
                        xs.dispose();
                        ys.dispose()
                    });
                    console.log('iteration: ', i, '--- train loss: ', loss);
                };
    
                logs = await evaluate(model, evalDs, config);
                
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
            };
       }