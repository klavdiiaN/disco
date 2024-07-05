import { Range, Repeat } from 'immutable'
import fs from 'node:fs/promises'
import path from 'node:path'

import type { Task, data } from '@epfml/discojs'
import { NodeImageLoader, NodeTabularLoader } from '@epfml/discojs-node'

async function simplefaceData (task: Task): Promise<data.DataSplit> {
  const dir = '../datasets/simple_face/'
  const youngFolder = dir + 'child/'
  const adultFolder = dir + 'adult/'

  const youngFiles = (await fs.readdir(youngFolder)).map(file => path.join(youngFolder, file))
  const adultFiles = (await fs.readdir(adultFolder)).map(file => path.join(adultFolder, file))
  const images = youngFiles.concat(adultFiles)

  const youngLabels = youngFiles.map(_ => 'child')
  const oldLabels = adultFiles.map(_ => 'adult')
  const labels = youngLabels.concat(oldLabels)
  return await new NodeImageLoader(task).loadAll(images, { labels })
}

async function cifar10Data (cifar10: Task): Promise<data.DataSplit> {
  const dir = '../datasets/CIFAR10/'
  const files = (await fs.readdir(dir)).map((file) => path.join(dir, file))
  const labels = Repeat('airplane', 24).toArray() // TODO read labels in csv
  return await new NodeImageLoader(cifar10).loadAll(files, { labels })
}

async function lusCovidData (lusCovid: Task): Promise<data.DataSplit> {
  const dir = '../datasets/lus_covid/'
  const covid_pos = dir + 'COVID+'
  const covid_neg = dir + 'COVID-'
  const files_pos = (await fs.readdir(covid_pos)).map(file => path.join(covid_pos, file))
  const label_pos = Range(0, files_pos.length).map(_ => 'COVID-Positive')

  const files_neg = (await fs.readdir(covid_neg)).map(file => path.join(covid_neg, file))
  const label_neg = Range(0, files_neg.length).map(_ => 'COVID-Negative')
  
  const files = files_pos.concat(files_neg)
  const labels = label_pos.concat(label_neg).toArray()

  const dataConfig = { labels, shuffle: true, validationSplit: 0.1, channels: 3 }
  return await new NodeImageLoader(lusCovid).loadAll(files, dataConfig)
}

async function titanicData (titanic: Task): Promise<data.DataSplit> {
  const dir = '../datasets/titanic_train.csv'

  const data = await (new NodeTabularLoader(titanic, ',').loadAll(
    ['file://'.concat(dir)],
    {
      features: titanic.trainingInformation?.inputColumns,
      labels: titanic.trainingInformation?.outputColumns,
      shuffle: false
    }
  ))
  return data
}

// dir should contain separate folders for train, push (optional) and validation (optional) data 

async function ppnetData (task: Task, dir: string): Promise<data.DataSplit> {
  dir = dir + '/train/';
  const labels = await fs.readdir(dir);
  if (task.trainingInformation !== undefined) {
    task.trainingInformation.LABEL_LIST = labels;
    task.trainingInformation.numClasses = labels.length} // update the number of classes parameter based on the dataset structure
  
  let allFiles: string[] = []
  let allLabels: string[] = []

  for (const label of labels){
    const dirImage = path.join(dir, label)
    const fileImage = (await fs.readdir(dirImage)).map(file => path.join(dirImage, file))
    allFiles = allFiles.concat(fileImage)
    const imageLabels = fileImage.map(_ => `${label}`)
    allLabels = allLabels.concat(imageLabels)
  }

  const trainingData = await new NodeImageLoader(task).loadAll(allFiles, {labels: allLabels, validationSplit: 0.1, channels: 3, shuffle: true})
  return trainingData
}

async function ppnetDataPush (task: Task, dir: string): Promise<data.DataSplit> {
  dir = dir + '/push/';
  const labels = await fs.readdir(dir);
  if (task.trainingInformation !== undefined) {
    task.trainingInformation.LABEL_LIST = labels;
    task.trainingInformation.numClasses = labels.length}
  
  let allFiles: string[] = []
  let allLabels: string[] = []

  for (const label of labels){
    const dirImage = path.join(dir, label)
    const fileImage = (await fs.readdir(dirImage)).map(file => path.join(dirImage, file))
    allFiles = allFiles.concat(fileImage)
    const imageLabels = fileImage.map(_ => `${label}`)
    allLabels = allLabels.concat(imageLabels)
  }

  const pushData = await new NodeImageLoader(task).loadAll(allFiles, {labels: allLabels, validationSplit: 0, channels: 3, shuffle: true}) //, { labels: trainingLabels })
  return pushData
}

async function ppnetDataVal (task: Task, dir: string): Promise<data.DataSplit> {
  dir = dir + '/validation/';
  const labels = await fs.readdir(dir);
  if (task.trainingInformation !== undefined) {
    task.trainingInformation.LABEL_LIST = labels;
    task.trainingInformation.numClasses = labels.length}
  
  let allFiles: string[] = []
  let allLabels: string[] = []

  for (const label of labels){
    const dirImage = path.join(dir, label)
    const fileImage = (await fs.readdir(dirImage)).map(file => path.join(dirImage, file))
    allFiles = allFiles.concat(fileImage)
    const imageLabels = fileImage.map(_ => `${label}`)
    allLabels = allLabels.concat(imageLabels)
  }

  const pushData = await new NodeImageLoader(task).loadAll(allFiles, {labels: allLabels, validationSplit: 0, channels: 3, shuffle: true}) //, { labels: trainingLabels })
  return pushData
}

export async function getTaskData (task: Task): Promise<data.DataSplit> {
  switch (task.id) {
    case 'simple_face':
      return await simplefaceData(task)
    case 'titanic':
      return await titanicData(task)
    case 'cifar10':
      return await cifar10Data(task)
    case 'lus_covid':
      return await lusCovidData(task)
    case 'YOUR CUSTOM TASK HERE':
      throw new Error('YOUR CUSTOM FUNCTION HERE')
    default:
      throw new Error(`Data loader for ${task.id} not implemented.`)
  }
}

export async function getPpnetData (task: Task, dir: string): Promise<data.DataSplit> {
  return await ppnetData(task, dir)
}

export async function getPpnetDataPush (task: Task, dir: string): Promise<data.DataSplit> {
  return await ppnetDataPush(task, dir)
}

export async function getPpnetDataVal (task: Task, dir: string): Promise<data.DataSplit> {
  return await ppnetDataVal(task, dir)
}
