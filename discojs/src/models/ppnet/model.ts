import * as tf from '@tensorflow/tfjs';
import type { ppnetConfig } from './config.js';
import { DEFAULT_CONFIG } from './config.js';
import { train } from './train.js'; // implement
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types.js'; // find
import type { TrainingCallbacks } from './types.js';
//import type { Dataset } from '../dataset/index.js'


// SEPARATE FUNCTIONS //
async function mobileNetv2 (
    pretrainedPath: string
): Promise<tf.LayersModel> {
    //const handler = tf.io.fileSystem(pretrainedPath);
    const mobileNetv2 = await tf.loadLayersModel(pretrainedPath);
    //console.log(mobileNetv2.summary());
    const outputLayer = mobileNetv2.getLayer('out_relu').output;
    //const outputLayer = mobileNetv2.output;
    //console.log('Output kayer: ', outputLayer)
    const featureExtractor = tf.model({name: 'features', inputs: mobileNetv2.inputs, outputs: outputLayer});

    return featureExtractor;
};

/*async function convFeatures (cfg: ppnetConfig): Promise<tf.LayersModel> {
    const config = Object.assign(
        {
            name: 'features',
        }, 
        cfg
    );
    const inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
    //console.log('Input shape:', inputs.shape);
    let x: tf.Tensor | tf.SymbolicTensor | tf.Tensor[] | tf.SymbolicTensor[];

    x = (await mobileNetv2(config.pretrainedPath)).apply(inputs) as tf.SymbolicTensor;
    //console.log('Output shape:', x.shape)
   
    return tf.model({ name: config.name, inputs: inputs, outputs: x });
};*/

/*async function addOnLayers (features: tf.LayersModel, cfg: ppnetConfig): Promise<tf.LayersModel>{
    const config = Object.assign(
        {
            name: 'addOnLayers',
        }, 
        cfg
    ); 

    const inputs = features.output;
    //let x = inputs
    let x: tf.Tensor | tf.SymbolicTensor | tf.Tensor[] | tf.SymbolicTensor[];
    //x = cnnFeatures;
    x = inputs;

    x = tf.layers.conv2d({
        name: 'add_on_layer/conv2_1',
        filters: config.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'relu'
    }).apply(x);
    
    x = tf.layers.conv2d({
        name: 'add_on_layer/conv2d_2',
        filters: config.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'sigmoid'
    }).apply(x) as tf.SymbolicTensor;

    return tf.model({ name: config.name, inputs: features.inputs, outputs: x });
};*/

/*async function convLayers(cfg: ppnetConfig) {
    const featuresModel = await convFeatures(cfg);
    featuresModel.trainable = false;
    console.log(featuresModel.summary());
    const convLayers = await addOnLayers(featuresModel, cfg);
    return convLayers;
}*/

async function convLayers(cfg: ppnetConfig, inputs: tf.SymbolicTensor) {
    //const baseModel = await mobileNetv2(cfg.pretrainedPath);
    //const inputs = tf.input({shape: [cfg.imgSize, cfg.imgSize, 3]});
    //let x = baseModel.apply(inputs, {name: 'features'}) as tf.SymbolicTensor;

    // Add additional convolutional layers with explicit names
    let x = tf.layers.conv2d({
        filters: cfg.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'relu',
        name: 'additional_conv_1'
    }).apply(inputs) as tf.SymbolicTensor;

    x = tf.layers.conv2d({
        filters: cfg.prototypeShape[3],
        kernelSize: 1,
        kernelInitializer: 'glorotUniform',
        activation: 'sigmoid',
        name: 'additional_conv_2'
    }).apply(x) as tf.SymbolicTensor;

    //const finalModel = tf.model({inputs: inputs, outputs: x, name: 'featureLayers'});

    // Create and return the complete model
    return x
    //{ model: finalModel, features: baseModel }
};

// LAYERS //
class L2Convolution_ extends tf.layers.Layer {
    private config: Object;
    private prototypeShape: number[];
    private featureShape: number[];

    private prototypeVectors!: tf.LayerVariable;
    private ones!: tf.LayerVariable;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'l2_convolution' }, config); 
        this.name = 'l2_convolution';
        this.prototypeShape = [config.prototypeShape[1], config.prototypeShape[2], config.prototypeShape[3], config.prototypeShape[0]];
        this.featureShape = config.featureShape;
    }

    build(inputShape: tf.Shape | tf.Shape[]): void {
        this.prototypeVectors = this.addWeight('proto_vec', this.prototypeShape, 'float32', tf.initializers.randomUniform({ minval: 0, maxval: 1 }));
        this.ones = this.addWeight('ones', this.prototypeShape, 'float32', tf.initializers.ones(), undefined, false);
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [null, this.featureShape[0], this.featureShape[1], this.prototypeShape[3]];
    }

    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return Object.assign({}, config, this.config);
    }

    get protVectors(){
        return this.prototypeVectors.read()
    };

    // implement //
    set protVectors(update: tf.Tensor){
        this.prototypeVectors.write(update);
    };

    freezePrototypes(){
        this.prototypeVectors.trainable = false
    };

    unfreezePrototypes(){
        this.prototypeVectors.trainable = true
    };

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);

            // B = batchSize, P = prototype, D = dimension, N = number
            const x2 = tf.square(inputs) as tf.Tensor4D;    // [B, 7, 7, PD]
            const x2_patch_sum = tf.conv2d(
                x2,
                this.ones.read() as tf.Tensor4D,
                1,
                'valid'
            );                                              // [B, 7, 7, PN]

            let p2 = tf.square(this.prototypeVectors.read());
            p2 = tf.sum(p2, [0, 1, 2], false);
            p2 = tf.reshape(p2, [1, 1, -1]);                // [PN]

            let xp = tf.conv2d(
                inputs as tf.Tensor4D,
                this.prototypeVectors.read() as tf.Tensor4D,
                1,
                'valid'
            );
            xp = tf.mul(xp, tf.scalar(-2));                 // [B, 7, 7, PN]

            const intermediate_result = tf.add(xp, p2);
            const distances = tf.relu(tf.add(x2_patch_sum, intermediate_result));

            return distances;
        })
    }

    static get className(): string {
        return 'L2Convolution';
    }
};
tf.serialization.registerClass(L2Convolution_);
const L2Convolution = (config: any) => new L2Convolution_(config);

class MinDistancesPooling_ extends tf.layers.Layer {
    private config: Object;
    private kernelSize: [number, number];
    private numPrototypes: number;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'min_distances' }, config); 
        this.name = 'min_distances';
        this.kernelSize = [config.featureShape[0], config.featureShape[1]];
        this.numPrototypes = config.prototypeShape[0];
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return [null, this.numPrototypes];
    }
    
    getConfig(): tf.serialization.ConfigDict {
        const config = super.getConfig();
        return Object.assign({}, config, this.config);
    }

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);

            let distances = tf.mul(inputs, tf.scalar(-1)) as tf.Tensor4D;
            let minDistances = tf.pool(
                distances,
                this.kernelSize,
                'max',
                'valid'
            ) as tf.Tensor;
            minDistances = tf.mul(minDistances, tf.scalar(-1));                 // [B, 1, 1, PN]
            minDistances = tf.reshape(minDistances, [-1, this.numPrototypes])   // [B, PN]

            return minDistances;
        })
    }

    static get className(): string {
        return 'MinDistancesPooling';
    }
};
tf.serialization.registerClass(MinDistancesPooling_);
const MinDistancesPooling = (config: any) => new MinDistancesPooling_(config);

class Distance2Similarity_ extends tf.layers.Layer {
    private config: Object;
    private epsilon: number;
    //private prototypeActivationFunction: string;

    constructor(config: any) {
        super(config);

        this.config = Object.assign({ name: 'distance_to_similarity' }, config); 
        this.name = 'distance_to_similarity';
        //this.prototypeActivationFunction = config.prototypeActivationFunction;
        this.epsilon = 1e-4;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }

    call(inputs: tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[], kwargs: Kwargs): tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[] {
        return tf.tidy(() => {
            if (Array.isArray(inputs)) {
                inputs = inputs[0];
            }
            this.invokeCallHook(inputs, kwargs);
            return tf.log(
                tf.div(
                    tf.add(inputs, tf.scalar(1)),
                    tf.add(inputs, tf.scalar(this.epsilon))
                ) 
            )
        })
    }

    static get className(): string {
        return 'Distance2Similarity';
    }
};
tf.serialization.registerClass(Distance2Similarity_);
const Distance2Similarity = (config: any) => new Distance2Similarity_(config);

async function PPNet (conf: ppnetConfig): Promise<tf.LayersModel> {
    const configDefaults = {
        name: 'prototypical_part_network',
        ...DEFAULT_CONFIG
      }

    const config = Object.assign({}, configDefaults, conf);
    const inputs = tf.input({ shape: [config.imgSize, config.imgSize, 3] });
    const baseModel = await mobileNetv2(conf.pretrainedPath);
    const x = baseModel.apply(inputs) as tf.SymbolicTensor;

    const convFeatures = await convLayers(conf, x);
    //console.log(convFeatures.summary());
    //const baseModel = convFeatures.features;
    //const convOutput = convFeatures.apply(inputs);
    //console.log('Additional conv layers output shape:', convOutput.shape);
    const distances = L2Convolution(conf).apply(convFeatures) as tf.SymbolicTensor;
    //console.log('Distances shape:', distances.shape);

    const minDistances = MinDistancesPooling(conf).apply(distances) as tf.SymbolicTensor;
    //console.log('Min distances shape:', minDistances.shape);

    const prototype_activations = Distance2Similarity(conf).apply(minDistances);
    //console.log('Prototypes activations:', prototype_activations)

    const logits = tf.layers.dense({
        name: 'logits',
        units: config.numClasses
    }).apply(prototype_activations) as tf.SymbolicTensor;

    const combinedOutput = tf.layers.concatenate({axis: -1}).apply([logits, minDistances]) as tf.SymbolicTensor;
    //console.log('Output:', combinedOutput.shape);
    const model = tf.model({ name: 'PPNet_final', inputs: inputs, outputs: [combinedOutput, distances, convFeatures]});
    //console.log(model.summary())
    return model;
};

// ask about this //
/**
 * tfjs does not export LazyIterator and Dataset...
 */
declare abstract class LazyIterator<T> {
    abstract next (): Promise<IteratorResult<T>>
  }
  
declare abstract class Dataset<T> {
    abstract iterator (): Promise<LazyIterator<T>>
    size: number
};

// implement
/*class PPNetModel extends tf.LayersModel {
    protected peakMemory: { value: number}
    constructor(protected readonly config: ppnetConfig) {
        let ppnet: tf.LayersModel;
    
        super({ inputs: [], outputs: [] }); // Call super() first
        this.peakMemory = {value: 0};
        PPNet(config)
            .then(output => {
                ppnet = output;
                const { inputs, outputs } = ppnet;
                this.inputs = inputs;
                this.outputs = outputs;
                this.peakMemory.value = 0;
                Object.assign(this, ppnet); // Assign properties after super() call
            })
            .catch(error => {
                console.log(error);
            });
    };*/
    class PPNetModel extends tf.LayersModel {
        protected peakMemory: { value: number };
    
        private constructor(model: tf.LayersModel, protected readonly config: ppnetConfig) {
            super({ inputs: model.inputs, outputs: model.outputs });
            this.peakMemory = { value: 0 };
            Object.assign(this, model);
        }
    
        static async createInstance(config: ppnetConfig): Promise<PPNetModel> {
            try {
                const ppnet = await PPNet(config);
                return new PPNetModel(ppnet, config);
            } catch (error) {
                console.error(error);
                throw new Error("Failed to initialize PPNetModel");
            }
        };
    
    get getPPNetConfig() {
        return this.config
      }

    adjustNumClasses() {
        const { numClasses } = this.config;
        // Adjust model parameters based on the number of classes
    }

    async fitDataset<T> (
      dataset: Dataset<T>,
      args: tf.ModelFitDatasetArgs<T>,
      clientNumber?: number,
      pushDataset?: Dataset<T>,
    ): Promise<tf.History> {
      const config = { ...this.config, ...args }
  
      await train(
        this,
        dataset as tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
        config,
        args.epochs,
        args.callbacks as TrainingCallbacks,
        args.validationData as tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>,
        clientNumber,
        pushDataset as tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>
      )
  
      return new tf.History()
    }
  };

export {PPNetModel, L2Convolution_}
