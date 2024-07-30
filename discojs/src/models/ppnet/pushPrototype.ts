// This script implements prototypes projection to the nearest latent patch //
// Adapted from https://github.com/cfchen-duke/ProtoPNet 

/* 
-- Feed the image through the convolutional layers
-- Compute the distance between each prototype and each latent patch
-- Update the prototypes with the values of the their nearest patches
-- Save the prototypes as the most activated regions in the images using receptive field information of the nearest latent patches to these prototypes
*/

import { L2Convolution_} from './model.js';
import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs';
import * as path from 'path';
import { promisify } from 'util';
import { getProtoClassIdx } from './utils.js'
import { ppnetConfig } from './config.js';
import { computeRfPrototype, computeProtoLayerRfInfo } from './receptiveField.js'

const mkdir = promisify(fs.mkdir);
const writeFile = promisify(fs.writeFile);

function extractConvInfo(model: tf.LayersModel) {
    let filterSizes: Array<any> = [];
    let strides: Array<any> = [];
    let paddings: Array<any> = [];

    // Loop through each layer in the model
    model.layers.forEach(layer => {
    // Check if the layer is a convolutional layer
    if (layer.getClassName() === 'Conv2D') {
        // Extract the configuration of the layer
        const config = layer.getConfig();

        // Extract and save specific properties
        filterSizes.push(config.kernelSize);
        strides.push(config.strides);
        paddings.push(config.padding)
    }
})
    return [filterSizes, strides, paddings]
};

/**
 * This function performs the projection of prototypes on their nearest latent patches for visualization. 
 * Optionally, it also updates the prototypes with the values of their nearest latent patches.
 * 
 * @param cfg - model config
 * @param dataset - training or push dataset
 * @param model - trained model as tf.LayersModel
 * @param preprocessInputFunction - function to preprocess the images (in DISCO, data is preprocessed automatically so this parameter may be removed) 
 * @param prototypeLayerStride -  value of the stride of the prototype layer (coming from the receptive field information)
 * @param rootDirForSavingProts - directory to save prototypes
 * @param epochNumber - current epoch number
 * @param protImgFilePrefix - prefix to name saved prototypes
 * @param protSelfActFilePrefix - prefix to name saved self activations of the prototypes (a.k.a. heatmaps)
 * @param protBoundBoxFilePrefix - prefix to name saved images with prototypes as rectangular regions
 * @param saveProtClassId - boolean to indicate if we need to save prototypes class identity
 * @param update - boolean to indicate if we need to update the prototypes with the values of their nearest latent patches
 * @param clientNumber - index number of the currently active client (necessary for saving prototypes in the client-corresponding folder) or a string 'Local' for local training
 */

async function pushPrototypes (
    cfg: any,
    dataset: tf.data.Dataset<{ xs: tf.Tensor3D, ys: tf.Tensor1D }>, 
    model: tf.LayersModel,
    preprocessInputFunction: Function | null = null,
    prototypeLayerStride: number=1, 
    rootDirForSavingProts: string | null=null,
    epochNumber: number | null=null, 
    protImgFilePrefix: string | null=null,
    protSelfActFilePrefix: string, 
    protBoundBoxFilePrefix: string | null=null, 
    saveProtClassId: boolean = true,
    update: boolean = true,
    clientNumber: number | string | undefined=0): Promise<void> {

    const config = Object.assign({}, cfg);
    const numPrototypes = config.prototypeShape[0];
    const prototypeShape = config.prototypeShape;
    const numClasses = config.numClasses;
    // initialize an array of minimum distances
    let globalMinProtoDist: number[] = new Array(numPrototypes).fill(Infinity);
    // initialize an array of patches corresponding to minimum distances
    let globalMinFmapPatches = Array.from({length: numPrototypes}, () =>
        Array.from({length: prototypeShape[1]}, () =>
          Array.from({length: prototypeShape[2]}, () =>
            new Array(prototypeShape[3]).fill(0)
          )
        )
      );

    // Initialize protoRfBoxes (receptive field boxes) and protoBoundBoxes (actual bounding boxes) based on class identity requirement (if it's saved or not)
    let protoRfBoxes: number[][], protoBoundBoxes: number[][];
    if (saveProtClassId) { 
        protoRfBoxes = Array.from({length: numPrototypes}, () => Array(6).fill(-1));
        protoBoundBoxes = Array.from({length: numPrototypes}, () => Array(6).fill(-1));
    } else {
    protoRfBoxes = Array.from({length: numPrototypes}, () => Array(5).fill(-1));
    protoBoundBoxes = Array.from({length: numPrototypes}, () => Array(5).fill(-1));
    }
    let protoEpochDir = rootDirForSavingProts;
    if (rootDirForSavingProts) {
        if (epochNumber) {
        protoEpochDir = path.join(rootDirForSavingProts, `prots-client${clientNumber}/epoch-${epochNumber}`); //
        await mkdir(protoEpochDir, { recursive: true });
        }
    }
    let pushIter = 0;
    let searchBatchSize = config.batchSizePush;

    // compute conv layer receptive field information
    const [filterSizes, strides, paddings] = extractConvInfo(model);
    // compute receptive field information
    const protoLRfInfo = computeProtoLayerRfInfo(
        config.imgSize,
        filterSizes,
        strides,
        paddings,
        config.prototypeShape[2]
    );
    //console.log('RF info:', protoLRfInfo);

    // iterate over batches searching for the nearest latent patch to a prototype
    await dataset.forEachAsync((batch) => {
        try{
        const searchBatchInput = batch.xs;
        const labels = batch.ys;
        const searchY = labels.argMax(1); // because labels are one-hot encoded
        let startIndexOfSearchBatch = pushIter * searchBatchSize; // we need to keep track of all images in the push dataset to find one which has the nearest latent patch
        console.log('Start prototype updating')

        updatePrototypeOnBatch(config, 
            searchBatchInput,
            model, 
            startIndexOfSearchBatch,
            globalMinProtoDist, 
            globalMinFmapPatches,
            protoRfBoxes, 
            protoBoundBoxes,
            searchY, 
            numClasses,
            preprocessInputFunction, 
            prototypeLayerStride,
            protoEpochDir, 
            protImgFilePrefix,
            protSelfActFilePrefix,
            protoLRfInfo, 
            1e-4);
            pushIter += 1;
            console.log('Batch processing complete');
        } catch (error) {
            console.error('Error processing batch:', error);
        }
    });
    console.log('Completed all batches');
        
    // After updating prototypes, optionally save them
    if (protoEpochDir !== null && protBoundBoxFilePrefix !== null) {
        await writeFile(path.join(protoEpochDir, `${protBoundBoxFilePrefix}-receptive_field${epochNumber}.json`), JSON.stringify(protoRfBoxes));
        await writeFile(path.join(protoEpochDir, `${protBoundBoxFilePrefix}${epochNumber}.json`), JSON.stringify(protoBoundBoxes));
};

//console.log('globalMinFmapPatches after update:', globalMinFmapPatches[0][0][0])
  console.log('\tExecuting push ...');
  if (update) { // replace the prototypes with the values of the their nearest latent patches
    const globalMinFmapPatchesTensor = tf.tensor(globalMinFmapPatches);
    //console.log('globalMinFmapPatchesTensor:', globalMinFmapPatchesTensor)
    const prototypeUpdate: tf.Tensor = globalMinFmapPatchesTensor.reshape([prototypeShape[1], prototypeShape[2], prototypeShape[3], numPrototypes]);
    //console.log('update:', prototypeUpdate)
    const convLayer = model.getLayer('l2_convolution')
    if (convLayer instanceof L2Convolution_) {
        convLayer.protVectors = prototypeUpdate;
    }
  }
  console.log('prototypes updated')
};

/**
 * This function performs the search for the nearest latent patches and saves them as the prototypes.
 * 
 * @param cfg - model config
 * @param searchBatchInput - data batch to search for the nearest latent patch
 * @param model - trained model as tf.LayersModel
 * @param startIndexOfSearchBatch -  the current batch starting image's index in the entire push dataset
 * @param globalMinProtoDist - current smallest distances
 * @param globalMinFmapPatches - patches corresponding to the current smallest distances
 * @param protoRfBoxes - receptive field boxes of the prototypes 
 * @param protoBoundBoxes - bounding boxes of the prototypes
 * @param searchLabel - labels of the images in the current batch
 * @param numClasses - number of classes
 * @param preprocessInputFunction - function to preprocess the images (in DISCO, data is preprocessed automatically so this parameter may be removed) 
 * @param prototypeLayerStride - value of the stride of the prototype layer (coming from the receptive field information)
 * @param dirForSavingProts - directory to save prototypes
 * @param protImgFilePrefix - prefix to name saved prototypes
 * @param protSelfActFilePrefix - prefix to name saved self activations of the prototypes (a.k.a. heatmaps)
 * @param protoLRfInfo - receptive field information
 * @param epsilon - a very small number to avoid devision by zero
 */

async function updatePrototypeOnBatch(
    cfg: ppnetConfig, 
    searchBatchInput: tf.Tensor,
    model: tf.LayersModel,
    startIndexOfSearchBatch: number = 0, 
    globalMinProtoDist: number[], 
    globalMinFmapPatches: number[][][][],
    protoRfBoxes: number[][], 
    protoBoundBoxes: number[][],
    searchLabel: tf.Tensor, 
    numClasses: number = 2,
    preprocessInputFunction: Function | null=null, 
    prototypeLayerStride: number = 1,
    dirForSavingProts: string | null=null, 
    protImgFilePrefix: string | null=null,
    protSelfActFilePrefix: string, 
    protoLRfInfo: number[], 
    epsilon: number = 1e-4){
        let searchBatch: tf.Tensor;
        const searchBatchInputClone = tf.clone(searchBatchInput);
        if (preprocessInputFunction != null){
            searchBatch = preprocessInputFunction(searchBatchInput)
        } else {searchBatch = searchBatchInput}

        // get the prediction for the batch and separate obtained distances and the output of the convolutional layers
        const output = model.predict(searchBatch) as tf.Tensor[];
        const protoDist = output[1];
        const protoLInput = output[2];
        //console.log('Distances:', protoDist)

        // convert tf.Tensor to an Array to simplify following computations 
        const dataY = await searchLabel.data()
        //console.log('dataY:', dataY);
        const searchY = Array.from(dataY);
        //console.log('searchY:', searchY)
        // initialize an array of indices of images in each class
        const classToImgIndexArray: number[][] = Array.from({ length: numClasses }, () => []);
        searchY.forEach((imgLabel, imgIndex) => {
            classToImgIndexArray[imgLabel].push(imgIndex);
        });
        //console.log('classToImgIndexArray:', classToImgIndexArray)

        const prototypeShape = cfg.prototypeShape;
        const numPrototypes = prototypeShape[0];
        const protoH = prototypeShape[1]; // height
        const protoW = prototypeShape[2]; // width

        // iterate over all prototypes and find the nearest latent patches in the current batch
        for (let j=0; j<numPrototypes; j++){
            const protoClassId = getProtoClassIdx(cfg).slice([j]); // get the class to which this prototype belongs (a one-hot encoded tensor) and convert it into a number
            const targetClassTensor = protoClassId.argMax(1);
            const targetClassData = await targetClassTensor.data();
            const targetClass = targetClassData[0];
            if (classToImgIndexArray[targetClass].length === 0){continue} // if there are no images of this class in this batch, move to the next prototype

            const indicesForTargetClass = tf.tensor1d(classToImgIndexArray[targetClass], 'int32'); // get indices of the images in the batch which belong to the prototype class

            // get the distances between the image patches in the batch and the current prototype
            const gathered = tf.gather(protoDist, indicesForTargetClass, 0);
            //console.log('Proto dist:', protoDist)
            //console.log('Gathered:', gathered);

            // Now, slice to keep only index j (current prototype) along the second dimension, and all elements along the third and fourth dimensions
            const begin = [0, 0, 0, j];
            const size = [-1, -1, -1, 1]; // -1 means all elements along that dimension

            const protoDistJ = gathered.slice(begin, size);
            //console.log('ProtoDistJ:', protoDistJ);

            const batchMinProtoDistJTensor = tf.min(protoDistJ); // get the minimum distance (a tensor) and convert it into a number
            const batchMinProtoDistJData = await batchMinProtoDistJTensor.data();
            let batchMinProtoDistJ = batchMinProtoDistJData[0];
            //console.log(batchMinProtoDistJ)

            if (batchMinProtoDistJ < globalMinProtoDist[j]){ 
                const flatIndexData = await protoDistJ.argMin().data();
                const flatIndex = flatIndexData[0];
                const shape = protoDistJ.shape;
                let batchArgminProtoDistJ = [];
                // some ChatGPT magic to replace np.unravel_index() method
                let residual = flatIndex;
                for (let i = 0; i < shape.length; i++) {
                    let stride = 1;
                    for (let j = i + 1; j < shape.length; j++) {
                        stride *= shape[j];
                    }
                    batchArgminProtoDistJ.push(Math.floor(residual / stride));
                    residual %= stride;
                }
                /*
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                */
                batchArgminProtoDistJ[0] = classToImgIndexArray[targetClass][batchArgminProtoDistJ[0]];
                //console.log('batchArgminProtoDistJ', batchArgminProtoDistJ[0])

                // retrieve the corresponding feature map patch
                const imgIndexInBatch = batchArgminProtoDistJ[0];
                //console.log('imgIndexInBatch:', imgIndexInBatch)
                const fmapHeightStartIndex = batchArgminProtoDistJ[1] * prototypeLayerStride;
                //console.log('start height:', fmapHeightStartIndex)
                const fmapHeightEndIndex = fmapHeightStartIndex + protoH;
                const fmapWidthStartIndex = batchArgminProtoDistJ[2] * prototypeLayerStride;
                const fmapWidthEndIndex = fmapWidthStartIndex + protoW;

                const sizeForHeight = fmapHeightEndIndex - fmapHeightStartIndex;
                const sizeForWidth = fmapWidthEndIndex - fmapWidthStartIndex;

                // We slice the nearest patch from the entire convolutional output (a tensor) and convert it into an array
                const batchMinFmapPatchJTensor = protoLInput.slice(
                    [imgIndexInBatch, fmapHeightStartIndex, fmapWidthStartIndex, 0],
                    [1, sizeForHeight, sizeForWidth, -1]
                );

                let batchMinFmapPatchJ;
                batchMinFmapPatchJTensor.array().then(array => {
                    batchMinFmapPatchJ = array as number[][][];
                    globalMinFmapPatches[j]  = batchMinFmapPatchJ;
                })
                
                globalMinProtoDist[j] = batchMinProtoDistJ; // update the currently nearset patch

                //get the receptive field boundary of the image patch that generates the representation
                let searchBatchShape: number = 224;
                if (searchBatch.shape[2] !== undefined){
                    searchBatchShape = searchBatch.shape[2];
                };
                
                const rfPrototypej = computeRfPrototype(searchBatchShape, batchArgminProtoDistJ, protoLRfInfo);
                //console.log('RF info:', rfPrototypej);
                
                //get the whole image
                let originalImgJ = searchBatchInputClone.slice([rfPrototypej[0], 0, 0, 0],[1, -1, -1, -1]);
                const normImage = originalImgJ.sub(originalImgJ.min())
                                  .div(originalImgJ.max().sub(originalImgJ.min()));
                originalImgJ = normImage.squeeze();

                //console.log('Original image:', originalImgJ)
                let originalImgSize: number = 224;
                if (originalImgJ.shape[1] !== undefined){
                    originalImgSize = originalImgJ.shape[1]
                };

                //crop out the receptive field
                // Calculate slice sizes
                const heightSize = rfPrototypej[2] - rfPrototypej[1]; // For the first dimension
                const widthSize = rfPrototypej[4] - rfPrototypej[3]; // For the second dimension

                // Perform the slicing
                // tf.slice() takes the tensor to slice, an array of starting indices, and an array of sizes for each dimension
                const rfImgJ = tf.slice(originalImgJ, [rfPrototypej[1], rfPrototypej[3], 0], [heightSize, widthSize, -1]); // we don't save it later but may consider it if needed

                //save the prototype receptive field information
                protoRfBoxes[j][0] = rfPrototypej[0] + startIndexOfSearchBatch;
                protoRfBoxes[j][1] = rfPrototypej[1];
                protoRfBoxes[j][2] = rfPrototypej[2];
                protoRfBoxes[j][3] = rfPrototypej[3];
                protoRfBoxes[j][4] = rfPrototypej[4];

                const numCols = protoRfBoxes[0].length;
                if (numCols === 6 && searchY != null){
                    protoRfBoxes[j][5] = searchY[rfPrototypej[0]]
                };

                //find the most activated region in the original image
                const height = protoDist.shape[1]!;
                const width = protoDist.shape[2]!;
                const protoDistImgJ = protoDist.slice([imgIndexInBatch, 0, 0, j], [1, height, width, 1]);
                
                let protoActImgJ: tf.Tensor;
                protoActImgJ = tf.log(tf.div(tf.add(protoDistImgJ, 1), tf.add(protoDistImgJ, epsilon))) // apply activation function to convert distances into similarity scores
                const protoActImgJSqueezed = protoActImgJ.squeeze();
                
                // upsample the most activated patch
                const upsampledActImgJ = tf.image.resizeBilinear(protoActImgJSqueezed.expandDims(2) as tf.Tensor3D | tf.Tensor4D, [originalImgSize, originalImgSize]);
                const upsampledActImgJSqueezed = upsampledActImgJ.squeeze([2]);

                // enclose the most activated patch in an image in a rectangle
                const protoBoundJ = await findHighActivationCrop(upsampledActImgJSqueezed);
                //console.log('protoBoundJ:', protoBoundJ)

                // crop out the image patch with high activation as a prototype image
                const protoImgJ = tf.tidy(() => {
                    // Calculate the size of the slice for each dimension
                    const sizeHeight = protoBoundJ[1] - protoBoundJ[0];
                    const sizeWidth = protoBoundJ[3] - protoBoundJ[2];
                    const sizeDepth = originalImgJ.shape[2]!; // Assuming the third dimension is channels and we want all channels
                  
                    // Perform the slice operation
                    return originalImgJ.slice([protoBoundJ[0], protoBoundJ[2], 0], [sizeHeight, sizeWidth, sizeDepth]);
                  });
                // save the prototype boundary (rectangular boundary of highly activated region)
                protoBoundBoxes[j][0] = protoRfBoxes[j][0];
                protoBoundBoxes[j][1] = protoBoundJ[0];
                protoBoundBoxes[j][2] = protoBoundJ[1];
                protoBoundBoxes[j][3] = protoBoundJ[2];
                protoBoundBoxes[j][4] = protoBoundJ[3];

                const nCols = protoBoundBoxes[0].length;
                if (nCols === 6 && searchY){
                    protoBoundBoxes[j][5] = searchY[rfPrototypej[0]]
                };

                if (dirForSavingProts){
                    if (protSelfActFilePrefix){
                        // save the numpy array of the prototype self activation
                        const serializedTensor = protoActImgJ.arraySync();
                        // Ensure the directory exists
                        fs.mkdirSync(dirForSavingProts, { recursive: true });

                        // Construct the file path
                        const filePath = path.join(dirForSavingProts, `${protSelfActFilePrefix}${j}.json`); // Using .json for simplicity

                        // Save serialized tensor to a file
                        fs.writeFileSync(filePath, JSON.stringify(serializedTensor));
                    }
                    if (protImgFilePrefix){
                        // save the whole image containing the prototype as png
                        // Convert the tensor to a Uint8Array

                        const scaledImg = originalImgJ.mul(255).toInt();
                        const imageData = await tf.node.encodeJpeg(scaledImg as tf.Tensor3D);  // Encode as PNG
                        fs.mkdirSync(dirForSavingProts, { recursive: true });  // Ensure directory exists
                        const filePath = path.join(dirForSavingProts, `${protImgFilePrefix}-original${j}.png`);
                        fs.writeFileSync(filePath, imageData);  // Save the image


                        //save the prototype image (highly activated region of the whole image)
                        const scaledProtData = protoImgJ.mul(255).toInt();
                        const protData = await tf.node.encodeJpeg(scaledProtData as tf.Tensor3D);
                        const filePathProt = path.join(dirForSavingProts, `${protImgFilePrefix}${j}.png`);
                        fs.writeFileSync(filePathProt, protData);

                        // overlay (upsampled) self activation on original image and save the result
                        // implemented by ChatGPT, may not work well :)
                        let rescaledActImgJ = upsampledActImgJ.sub(upsampledActImgJ.min()).div(upsampledActImgJ.max().sub(upsampledActImgJ.min()));
                        const heatmapTensor = await applyJetColorMap(rescaledActImgJ);
                        const overlayedOriginalImgJ = scaledImg.mul(tf.scalar(0.5)).add(heatmapTensor.mul(tf.scalar(0.3)));
                        const overlayData = await tf.node.encodePng(overlayedOriginalImgJ as tf.Tensor3D);
                        const filePathOverlay = path.join(dirForSavingProts, `${protImgFilePrefix}-original_with_self_act${j}.png`);
                        fs.writeFileSync(filePathOverlay, overlayData);
                    }
                }     
            }
        }
    };

/**
 * This function encloses the most activated region in an image into a rectangle indicating a prototypical part.
 * 
 * @param activationMap - a tensor with similarity scores between image patches and a prototype
 * @param percentile - an minimum intensity percentile to enclose in a rectangle
 * 
 */
export async function findHighActivationCrop (activationMap: tf.Tensor, percentile: number = 95): Promise<number[]> {
    if (activationMap.rank !== 2) {
        throw new Error('Expected a 2D tensor');
    }
    const shapeMap = activationMap.shape as [number, number];

    const values = await activationMap.data();
    const sortedValues = Array.from(values).sort((a, b) => a - b);
    const thresholdIndex = Math.floor(percentile / 100 * sortedValues.length);
    const threshold = sortedValues[thresholdIndex];

    // Convert the activation map to a binary mask based on the threshold
    const maskArray = Array.from(values).map(value => value >= threshold ? 1 : 0);

    // Reshape the flat mask array back to the original shape of activationMap
    const mask2DArray: number[][] = [];
    for (let i = 0; i < shapeMap[0]; i++) {
        mask2DArray.push(maskArray.slice(i * shapeMap[1], (i + 1) * shapeMap[1]));
    }

    // Now, find the bounding box within the mask2DArray
    // Initialize coordinates
    let lowerY = 0, upperY = 0, lowerX = 0, upperX = 0;
    
    // Find lowerY
    lowerY = mask2DArray.findIndex(row => row.includes(1));
    // Find upperY
    upperY = mask2DArray.length - 1 - mask2DArray.slice().reverse().findIndex(row => row.includes(1));

    // Find lowerX and upperX by checking columns
    const transposedMask = mask2DArray[0].map((_: number, colIndex: number) => mask2DArray.map(row => row[colIndex]));
    //console.log('transposed mask:', transposedMask);
    lowerX = transposedMask.findIndex((col: Array<number>) => col.some(value => value > 0.5));
    upperX = transposedMask.length - 1 - transposedMask.slice().reverse().findIndex((col: Array<number>) => col.some(value => value > 0.5));

    // Adjust the coordinates as needed
    return [lowerY, upperY + 1, lowerX, upperX + 1];
};

// colors encoding for heatmaps //
interface JetColor {
    value: number;
    color: [number, number, number]; // Tuple type for RGB colors
}

const jetColors: JetColor[] = [
    { value: 0, color: [0, 0, 128] }, // Dark Blue
    { value: 0.35, color: [0, 0, 255] }, // Blue
    { value: 0.5, color: [0, 255, 255] }, // Cyan
    { value: 0.6, color: [255, 255, 0] }, // Yellow
    { value: 0.8, color: [255, 0, 0] }, // Red
    { value: 1, color: [128, 0, 0] } // Dark Red
];

// the following two functions allow to visualize the most activated regions with color heatmaps, implemented by ChatGPT //
function interpolateColor(value: number): [number, number, number] {
    let startColor: JetColor = jetColors[0]; // Default to the first color
    let endColor: JetColor = jetColors[jetColors.length - 1]; // Default to the last color

    for (let i = 0; i < jetColors.length - 1; i++) {
        if (value >= jetColors[i].value && value < jetColors[i + 1].value) {
            startColor = jetColors[i];
            endColor = jetColors[i + 1];
            break;
        }
    }
    // Calculate the ratio for interpolation
    const ratio = (value - startColor.value) / (endColor.value - startColor.value);

    // Interpolate between startColor and endColor
    return startColor.color.map((startComponent, index) =>
        Math.round(startComponent + ratio * (endColor.color[index] - startComponent))
    ) as [number, number, number];
};

export async function applyJetColorMap(imageTensor: tf.Tensor) {
    // First, ensure the imageTensor is reshaped to a flat tensor if it's not already 1D.
    const flattenedImageTensor = imageTensor.reshape([-1]);

    // Now, properly await the array() method to get the actual array data.
    const dataArray = await flattenedImageTensor.array() as number[];

    // Given dataArray is an array of numbers, you can now map over it.
    // The following demonstrates mapping each value to an RGB color using a hypothetical interpolateColor function.
    const colorMappedFlatArray: number[] = dataArray.flatMap(normalizedValue => 
        interpolateColor(normalizedValue)
    );

    // Assuming interpolateColor returns an array of numbers [r, g, b] for each input value,
    // and flatMap is used to flatten the array of arrays into a single array.

    // Now create a 3D tensor from the flat array of RGB values.
    const [height, width] = imageTensor.shape;
    const colorMappedTensor = tf.tensor3d(colorMappedFlatArray, [height, width, 3], 'float32');

    return colorMappedTensor;
};

export { pushPrototypes }