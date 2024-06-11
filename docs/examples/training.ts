import * as path from 'node:path'
import '@tensorflow/tfjs-node'

import type { Image, Task, TypedDataset } from '@epfml/discojs'
import { Disco, fetchTasks, Dataset } from '@epfml/discojs'
import { loadCSV, loadImagesInDir } from '@epfml/discojs-node'
import { startServer } from 'server'
import { Repeat } from 'immutable'

/**
 * Example of discojs API, we load data, build the appropriate loggers, the disco object
 * and finally start training.
 */
async function runUser (url: URL, task: Task, dataset: TypedDataset): Promise<void> {
  // Create Disco object associated with the server url, the training scheme
  const disco = new Disco(task, { url, scheme: 'federated' })
  for await (const _ of disco.fit(dataset)); // Start training on the dataset

  // Stop training and disconnect from the remote server
  await disco.close()
}

async function main (): Promise<void> {
  // Arbitrary chosen Task ID
  const NAME: string = 'titanic'

  // Launch a server instance
  const [server, url] = await startServer()

  // Get all pre-defined tasks
  const tasks = await fetchTasks(url)

  // Choose the task and load local data
  // Make sure you first ran ./get_training_data
  let task: Task | undefined
  let dataset: TypedDataset
  switch (NAME) {
    case 'titanic': {
      task = tasks.get('titanic')
      if (task === undefined) { throw new Error('task not found') }
      dataset = ["tabular", loadCSV("../../datasets/titanic_train.csv")]
      break
    }
    case 'simple_face': {
      task = tasks.get('simple_face')
      if (task === undefined) { throw new Error('task not found') }
      dataset = ["image", await loadSimpleFaceData()]
      break
    }
    default:
      throw new Error('task id not found')
  }

  // Add more users to the list to simulate more than 3 clients
  await Promise.all([
    runUser(url, task, dataset),
    runUser(url, task, dataset),
    runUser(url, task, dataset)
  ])

  // Close server
  await new Promise((resolve, reject) => {
    server.once('close', resolve)
    server.close(reject)
  })
}

async function loadSimpleFaceData(): Promise<Dataset<[Image, string]>> {
  const folder = "../datasets/simple_face";

  const [adults, childs]: Dataset<[Image, string]>[] = [
    (await loadImagesInDir(path.join(folder, "adult"))).zip(Repeat("adult")),
    (await loadImagesInDir(path.join(folder, "child"))).zip(Repeat("child")),
  ];

  return adults.chain(childs);
}

// You can run this example with "npm start" from this folder
main().catch(console.error)
