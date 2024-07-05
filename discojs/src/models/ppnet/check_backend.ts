import * as tf from '@tensorflow/tfjs';

console.log('Available backends:', Object.keys(tf.engine().registryFactory));