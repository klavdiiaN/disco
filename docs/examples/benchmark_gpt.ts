import type { Task } from '@epfml/discojs-core'
import { fetchTasks, data, models } from '@epfml/discojs-core'
import { NodeTextLoader } from '@epfml/discojs-node'
import * as tf from '@tensorflow/tfjs'

async function main(): Promise<void> { 
  // Launch a server instance
  const url = new URL('http://localhost:8080')
  
  // Fetch the wikitext task from the server
  const tasks = await fetchTasks(url)
  const task = tasks.get('wikitext-103')
  if (task === undefined) { throw new Error('task not found') }
  // Load the wikitext dataset from the `datasets` folder
  
  // Toggle TRAIN_MODEL to either train and save a new model from scratch or load an existing model
  
  const config: models.GPTConfig = {
    modelType: 'gpt-nano',
    lr: 0.0001,
    maxIter: 5,
    evaluateEvery:10000,
    maxEvalBatches: 10,
    blockSize: 8,
    vocabSize: 50258
  }
  const modelType = 'gpt2'//['gpt-nano', 'gpt-micro', 'gpt-mini', 'gpt2']
  const contextLength = 2048 // [128, 256, 512, 1024, 2048]
  const batchSize = 32 //[8, 16, 32, 64]
  
  console.log(`Begin loop - Memory: ${(tf.memory().numBytes / 1024 / 1024).toFixed(2)} MB`, `Num tensors: ${tf.memory().numTensors}`)
  task.trainingInformation.batchSize = batchSize
  config.modelType = modelType as models.GPTModelType
  config.blockSize = contextLength
  console.log(`\tmodel type ${modelType} \n\tbatch size ${batchSize} \n\tcontext length ${contextLength}`)
  // Reload the dataset to batch it with the right batch size
  const dataset = await loadWikitextData(task)
  const preprocessedDataset = dataset.train.preprocess().batch().dataset
  const model = new models.GPT(config)
  const logGenerator = model.train(preprocessedDataset, undefined, 1) // 5 epochs
  for await (const logs of logGenerator) {
    const updateTime = logs.weightUpdateTime ?? 0
    const msPerToken = updateTime / batchSize / contextLength  
    console.log(`\t\t\t${msPerToken.toFixed(2)} ms/token <br> ${logs.memory?.toFixed(0)} MB`)
  }
  model.dispose()
}

async function loadWikitextData (task: Task): Promise<data.DataSplit> {
  const loader = new NodeTextLoader(task)
  const dataSplit: data.DataSplit = {
    train: await data.TextData.init(await loader.load('../../datasets/wikitext/wiki.train.tokens', {shuffle: true}), task),
    validation: await data.TextData.init(await loader.load('../../datasets/wikitext/wiki.valid.tokens', {shuffle: true}), task)
  }
  return dataSplit
}

// You can run this example with "npm start" from this folder
main().catch(console.error)