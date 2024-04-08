import type { Task } from '@epfml/discojs-core'
import {
  Disco, fetchTasks, data, client as clients,
  aggregator as aggregators, informant, serialization, models
} from '@epfml/discojs-core'
import { NodeTextLoader } from '@epfml/discojs-node'
import fs from 'node:fs'
import fsPromises from 'node:fs/promises'


async function main(): Promise<void> { 
  // Launch a server instance
  const url = new URL('http://localhost:8080')
  
  // Fetch the wikitext task from the server
  const tasks = await fetchTasks(url)
  const task = tasks.get('wikitext-103')
  if (task === undefined) { throw new Error('task not found') }
  
  const modelFolder = './models'
  const modelFileName = 'model_10000it.json'
  
  // Toggle TRAIN_MODEL to either train and save a new model from scratch or load an existing model
  const TRAIN_MODEL = false
  if (TRAIN_MODEL) {
    // Load the wikitext dataset from the `datasets` folder
    const dataset = await loadWikitextData(task)
  
    // Initialize a Disco instance and start training a language model
    const aggregator = new aggregators.MeanAggregator()
    const client = new clients.federated.FederatedClient(url, task, aggregator)
    const trainingInformant = new informant.FederatedInformant(task, 10)
    const disco = new Disco(task, { scheme: 'federated', client, aggregator, informant: trainingInformant })
    await disco.fit(dataset)
  
    // Get the model and complete the prompt
    if (aggregator.model === undefined) {
      throw new Error('model was not set')
    }
    const model = aggregator.model as models.GPT
    // Save the trained model
    try {
      if (!fs.existsSync(modelFolder)) {
        fs.mkdirSync(modelFolder)
      }
    } catch (err) {
      console.error(err);
    }
    const encoded = await serialization.model.encode(model)
    await fsPromises.writeFile(`${modelFolder}/${modelFileName}`, encoded)
    
    // Retrieve the tokenizer used during training
    const tokenizer = await models.getTaskTokenizer(task)
    const prompt = 'The game began development in 2010 , carrying over a large portion'
    const generations = await model.generate(prompt, tokenizer)
    console.log(generations)
    
    await disco.close()
  } else {
    // Load the trained model
    const content = await fsPromises.readFile(`${modelFolder}/${modelFileName}`)
    const model = await serialization.model.decode(content) as models.GPT

    // Retrieve the tokenizer used during training
    const tokenizer = await models.getTaskTokenizer(task)
    // The game began development in 2010 , carrying over a large portion
    const prompt = 'Hello world how'
    console.log(await model.generate(prompt, tokenizer, 20))
  }
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
