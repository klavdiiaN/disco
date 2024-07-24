import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs';
import sharp from 'sharp';
import Jimp from 'jimp';
import * as path from 'path';
import { readJsonFile } from './utils.js';
import { promisify } from 'util';
import { findHighActivationCrop, applyJetColorMap } from './pushPrototype.js';
import { parse } from 'ts-command-line-args'

// specify arguments to parse when running this script //
interface analysisArguments {
    testImgName: string
    testImgLabel: number
    clientNumber: number
    epochNumber: number
    numClasses: number
  }
  
  type BenchmarkAnalysisArguments = {
    [K in keyof analysisArguments as Exclude<K, 'task'>]: analysisArguments[K]
  } & {
    help?: boolean
  }

const args = parse<BenchmarkAnalysisArguments>(
    {
      testImgName: { type: String, alias: 'i', description: 'Name of an image to analyse'},
      testImgLabel: { type: Number, alias: 'l', description: 'Label of the test image'},
      clientNumber: { type: Number, alias: 'n', description: 'Index number of the client which model will be used for analysis, for local models use the default value', defaultValue: 0 },
      epochNumber: { type: Number, alias: 'e', description: 'Number of epochs used for training', defaultValue: 10 },
      numClasses: { type: Number, alias: 'c', description: 'Number of classes in the dataset', defaultValue: 2 },
      help: { type: Boolean, optional: true, alias: 'h', description: 'Prints this usage guide' }
    },
    {
      helpArg: 'help',
    }
  )

const mkdir = promisify(fs.mkdir);

// specify the test image to be analyzed //
const testImgDir = './';
const testImgName = args.testImgName;
const testImgLabel = args.testImgLabel;
const saveAnalysisPath = testImgDir;
const testImgPath = path.join(testImgDir, testImgName);

// load the model //
const loadModelDir = `./models-client${args.clientNumber}/epoch${args.epochNumber}-final/`
const loadModelName = 'model.json'
const loadModelPath = path.join(loadModelDir, loadModelName)

console.log('Load model from', loadModelPath)

const ppnet = await tf.loadLayersModel(`file://${loadModelPath}`)

const cfg = {
    imgSize: 224,
    numProt: 10*args.numClasses,
    numClasses: args.numClasses,
    prototypeShape: [10*args.numClasses, 1, 1, 128]
}
const config = Object.assign({}, cfg);

// SANITY CHECK //
// confirm prototype class identity
const loadProtDir = `./prots-client${args.clientNumber}/epoch-${args.epochNumber}/` // directory with saved prototypes
const protInfoPath = path.join(loadProtDir, `prot_bb${args.epochNumber}.json`)

const prototypeInfoJSON = fs.readFileSync(protInfoPath, 'utf-8');
const prototypeInfo = JSON.parse(prototypeInfoJSON);

let prototypeImgIdentity;
if (prototypeInfo) {
    prototypeImgIdentity = prototypeInfo.map((row: Array<number>) => row[row.length - 1]);
    //console.log('prototypeImgIdentity', prototypeImgIdentity )
    const uniqueElements = new Set(prototypeImgIdentity);
    //console.log('uniqueElements', uniqueElements)
    const numberOfUniqueElements = uniqueElements.size;
    console.log('Prototypes are chosen from ', numberOfUniqueElements, ' number of classes.');
    console.log('Their class identities are: ', prototypeImgIdentity)
} else {
    console.log("Failed to load or parse the JSON data.");
};

// confirm that a prototype connects most strongly to its own class //
const lastLayer = ppnet.getLayer('logits');
const lastLayerWeights = lastLayer.weights[0].read()
//console.log('lastLayerWeights: ', lastLayerWeights)
const protMaxConnection = tf.argMax(lastLayerWeights, 1)

let sum = 0
if (prototypeImgIdentity){
    for (let i=0; i<prototypeImgIdentity.length; i++){
        if (protMaxConnection === prototypeImgIdentity[i]){
            sum += 1
        }
    }
} else {console.log('WARNING: Prototype image identity is undefined.')}

if (sum === config.numProt){
    console.log('All prototypes connect most strongly to their respective classes.')
} else {
    console.log('WARNING: Not all prototypes connect most strongly to their respective classes.')
}

// HELPER FUNCTION FOR PLOTTING //

function saveProt(fileName: string, index: number): void{
    const imgPath = path.join(loadProtDir, `prot${index}.png`)
    sharp(imgPath).toFile(fileName).then(() => {
            console.log('Image saved successfully!');
        }).catch(err => {
            console.error('Error saving image:', err);
        });
};

function saveProtSelfAct(fileName: string, index: number): void{
    const imgPath = path.join(loadProtDir, `prot-original_with_self_act${index}.png`)
    sharp(imgPath).toFile(fileName).then(() => {
            console.log('Image saved successfully!');
        }).catch(err => {
            console.error('Error saving image:', err);
        });
};

async function saveProtOriginalWithBBox(fileName: string, index: number, bBoxHeightStart: number,
    bBoxHeightEnd: number, bBoxWidthStart: number, bBoxWidthEnd: number, color: number, thickness: number): Promise<void> {
        const imgPath = path.join(loadProtDir, `prot-original${index}.png`) 
        try {
            const imgBuffer = await sharp(imgPath).toBuffer();
            let image = await Jimp.read(imgBuffer);
            image = image.resize(config.imgSize, config.imgSize)
    
            // Draw the rectangle (from ChatGPT) //
            image.scan(bBoxWidthStart, bBoxHeightStart, bBoxWidthEnd - bBoxWidthStart, bBoxHeightEnd - bBoxHeightStart, function(x, y, idx) {
                // Edge detection to draw only the border of the rectangle with the specified thickness
                if (x <= bBoxWidthStart + thickness || x >= bBoxWidthEnd - thickness || y <= bBoxHeightStart + thickness || y >= bBoxHeightEnd - thickness) {
                    this.bitmap.data[idx + 0] = (color >> 16) & 255; // Red
                    this.bitmap.data[idx + 1] = (color >> 8) & 255;  // Green
                    this.bitmap.data[idx + 2] = color & 255;         // Blue
                    this.bitmap.data[idx + 3] = 255;                 // Alpha
                }
            });
            await image.writeAsync(fileName);
        console.log('Image saved successfully!');
    } catch (err) {
        console.error('Error processing image:', err);
    }
};

async function imsaveWithBBox (fileName: string, bBoxHeightStart: number,
    bBoxHeightEnd: number, bBoxWidthStart: number, bBoxWidthEnd: number, color: number, thickness: number
): Promise<void>{ 
        try {
            const imgBuffer = await sharp(testImgPath).toBuffer();
            let image = await Jimp.read(imgBuffer);
            image = image.resize(config.imgSize, config.imgSize)
            
            // Draw the rectangle (from ChatGPT) //
            image.scan(bBoxWidthStart, bBoxHeightStart, bBoxWidthEnd - bBoxWidthStart, bBoxHeightEnd - bBoxHeightStart, function(x, y, idx) {
                // Edge detection to draw only the border of the rectangle with the specified thickness
                if (x <= bBoxWidthStart + thickness || x >= bBoxWidthEnd - thickness || y <= bBoxHeightStart + thickness || y >= bBoxHeightEnd - thickness) {
                    this.bitmap.data[idx + 0] = (color >> 16) & 255; // Red
                    this.bitmap.data[idx + 1] = (color >> 8) & 255;  // Green
                    this.bitmap.data[idx + 2] = color & 255;         // Blue
                    this.bitmap.data[idx + 3] = 255;                 // Alpha
                }
            });
            await image.writeAsync(fileName);
        console.log('Image saved successfully!');
    } catch (err) {
        console.error('Error processing image:', err);
    }
};

// load the test image and forward it through the network //
async function preprocess(imgPath: string) {
    try {
        // Load the image with jimp
        const image = await Jimp.read(imgPath);
        // Create a buffer of the image data
        const buffer = await image.getBufferAsync(Jimp.MIME_JPEG);
        let tensor = tf.node.decodeImage(buffer, 3);

        // resize and normalize
        tensor = tensor.resizeBilinear([config.imgSize, config.imgSize]);
        tensor = tensor.div(tf.scalar(255));

        return tensor;
    } catch (error) {
        console.error('Failed to convert image:', error);
    }
};

const imgTest = await preprocess(testImgPath);
console.log(imgTest)
let imgVariable = tf.tensor4d([0], [1, 1, 1, 1]);
//console.log(imgVariable)
if (imgTest){
    imgVariable = tf.expandDims(imgTest, 0);
}
const labelTest = tf.tensor(testImgLabel);

const output = ppnet.predict(imgVariable) as tf.Tensor[];
const logits = output[0].slice([0, 0], [-1, config.numClasses])
const minDist = output[0].slice([0, cfg.numClasses], [-1, cfg.prototypeShape[0]])
//console.log('minDist: ', minDist)
const protoDist = output[1];

const epsilon = 1e-7;
const protActivations = tf.log(tf.div(tf.add(minDist, 1), tf.add(minDist, epsilon)));
const protActPatterns = tf.log(tf.div(tf.add(protoDist, 1), tf.add(protoDist, epsilon)));
console.log('protActPatterns: ', protActPatterns)

const argMaxResults = tf.argMax(logits, 1);
let tables: [number, number][] = [];

// Loop over each element in the logits tensor's first dimension //
for (let i = 0; i < logits.shape[0]; i++) {
    const maxIndex = await argMaxResults.data();
    const label = await labelTest.data();
    tables.push([maxIndex[i], label[i]]);
    console.log(`${i} ${tables[tables.length - 1]}`);
};

let idx = 0;
const predictedCls = tables[idx][0];
const correctCls = tables[idx][1];
console.log('Predicted: ', predictedCls)
console.log('Correct: ', correctCls);

// save original image, may be not necessary

// MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE //
const savePath = path.join(saveAnalysisPath, 'most_activated_prototypes')
await mkdir(savePath, { recursive: true });
console.log('Most activated 10 prototypes of this image:');
const protActivationsArray = await protActivations.data()
const sortedActivations = tf.topk(protActivationsArray, protActivationsArray.length, true);
const arrAct = await sortedActivations.values.data();
const actSortedIndices = await sortedActivations.indices.data();
for (let i=1; i<11; i++){
    console.log(`top ${i} activated prototype for this image:`);
    let savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
                                `top-${i}_activated_prototype.png`)
    saveProt(savePath, actSortedIndices[i])

    const bBoxHeightStart = prototypeInfo[actSortedIndices[i]][1];
    const bBoxHeightEnd = prototypeInfo[actSortedIndices[i]][2];
    const bBoxWidthStart = prototypeInfo[actSortedIndices[i]][3];
    const bBoxWidthEnd = prototypeInfo[actSortedIndices[i]][4]
    savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
        `top-${i}_activated_prototype_in_original_img.png`)
    saveProtOriginalWithBBox(savePath, actSortedIndices[i], bBoxHeightStart, bBoxHeightEnd,
        bBoxWidthStart, bBoxWidthEnd, 0, 2) // check color and thickness

    savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
        `top-${i}_activated_prototype_self_act.png`)
    saveProtSelfAct(savePath, actSortedIndices[i]);

    console.log(`prototype index: ${actSortedIndices[i]}`);
    console.log(`prototype class identity ${prototypeImgIdentity[actSortedIndices[i]]}`);
    const protMaxConnectionArray = await protMaxConnection.data();
    if (protMaxConnectionArray[actSortedIndices[i]] !== prototypeImgIdentity[actSortedIndices[i]]){
        console.log(`prototype connection identity: ${protMaxConnectionArray[actSortedIndices[i]]}`)
    }
    console.log(`activation value (similarity score): ${arrAct[i]}`);
    const lastLayerWeight = await tf.max(lastLayerWeights, 1).data();
    //console.log('lastLayerWeight: ', lastLayerWeight)
    console.log(`last layer connection with predicted class: ${lastLayerWeight[actSortedIndices[i]]}`);

    const actPattern = tf.slice(protActPatterns, [0, 0, 0, 0], [-1, -1, -1, 1]).squeeze([0]);
    const upsampledActPattern = actPattern.resizeBilinear([config.imgSize, config.imgSize]).squeeze([2]);
    //const upsampledActPatternData = await upsampledActPattern.data()
    //console.log('upsampledActPatternData: ', upsampledActPatternData)

    // show the most highly activated patch of the image by this prototype
    const highActPatchIndices = await findHighActivationCrop(upsampledActPattern);
    console.log('highActPatchIndices: ', highActPatchIndices)
    if (imgTest){
        const size = [
            highActPatchIndices[1] - highActPatchIndices[0],
            highActPatchIndices[3] - highActPatchIndices[2],
            imgTest.shape[2]];
        const highActPatch = imgTest.slice([highActPatchIndices[0], highActPatchIndices[2], 0], size);
        savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
                            `most_highly_activated_patch_by_top-${i}_prototype.png`);
        
        const scaledHighActPatch = highActPatch.mul(255).toInt();
        const highActPatchData = await tf.node.encodeJpeg(scaledHighActPatch as tf.Tensor3D);
        fs.writeFileSync(savePath, highActPatchData)                           
    };
    savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
                            `most_highly_activated_patch_in_original_img_by_top-${i}_prototype.png`);
    imsaveWithBBox(savePath, highActPatchIndices[0], highActPatchIndices[1], highActPatchIndices[2],
        highActPatchIndices[3], 0, 2);

    // show the image overlayed with prototype activation map
    let rescaledActPattern = tf.sub(upsampledActPattern, tf.min(upsampledActPattern));
    rescaledActPattern = tf.div(rescaledActPattern, tf.add(tf.max(rescaledActPattern), epsilon));
    //const rescaledActPatternData = await rescaledActPattern.data()
    //console.log('rescaledActPattern: ', rescaledActPatternData)
    const heatmap = await applyJetColorMap(rescaledActPattern);
    if (imgTest){
        const imgTestNorm = imgTest.mul(255).toInt();
        let overlayedOriginalImgJ = imgTestNorm.mul(tf.scalar(0.5)).add(heatmap.mul(tf.scalar(0.3)));
        const overlayData = await tf.node.encodePng(overlayedOriginalImgJ as tf.Tensor3D);
        savePath = path.join(saveAnalysisPath, 'most_activated_prototypes',
                            `prototype_activation_map_by_top-${i}_prototype.png`);
        fs.writeFileSync(savePath, overlayData);
    };

    console.log('----------------------------------------------------')
};

if (predictedCls === correctCls){
    console.log('Prediction is correct')
} else {console.log('Prediction is wrong')}


