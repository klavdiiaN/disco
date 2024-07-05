// This script is analogous to cli.ts but adapted to the PPNet implementation //

import { List, Range } from 'immutable'
import fs from 'node:fs/promises'
import path from 'node:path'

import type { data, RoundLogs, Task } from '@epfml/discojs'
import { Disco, aggregator as aggregators, client as clients } from '@epfml/discojs'
import { startServer } from 'server'

import { getPpnetData, getPpnetDataPush, getPpnetDataVal } from './data.js'
import { args } from './args.js'

async function runUser(
  task: Task,
  url: URL,
  data: data.DataSplit,
  clientNumber: number | undefined=0, // needed to save prototypes into client-specific folders
  dataPush?: data.DataSplit, // not modified training data, i.e. not augmented
  dataVal?: data.DataSplit   // separate validation set
): Promise<List<RoundLogs>> {
  const client = new clients.federated.FederatedClient(
    url,
    task,
    new aggregators.MeanAggregator(),
  );

  // force the federated scheme
  const disco = new Disco(task, { scheme: "federated", client });

  let logs = List<RoundLogs>();
  for await (const round of disco.fit(data, clientNumber, dataPush, dataVal)) logs = logs.push(round);

  await disco.close();
  return logs;
}

// parameter dirMain should contain separate data folders for each of collaborating clients 
async function main (task: Task, numberOfUsers: number, dirMain: string, numClasses?: number, push?: boolean, val?: boolean): Promise<void> {
  console.log(`Started federated training of ${task.id}`)
  console.log({ args })
  const [server, url] = await startServer(numClasses);

  let dataPaths: Array<string> = [];
  const files = await fs.readdir(dirMain);
  for (const file of files){
    const pathToFile = path.join(dirMain, file);
    dataPaths.push(pathToFile)
  };

  let dataAll: Array<data.DataSplit> = []; // training data for all clients
  for (let i=0; i<dataPaths.length; i++){
    const data = await getPpnetData(task, dataPaths[i]);
    dataAll.push(data);
  };

  let dataAllPush: Array<data.DataSplit> = []; // push data for all clients
  let dataAllVal: Array<data.DataSplit> = [];  // validation data for all clients

  if (push){
    for (let i=0; i<dataPaths.length; i++){
      const dataPush = await getPpnetDataPush(task, dataPaths[i]);
      dataAllPush.push(dataPush);
  };
}

  if (val){
    for (let i=0; i<dataPaths.length; i++){
      const dataVal = await getPpnetDataVal(task, dataPaths[i]);
      dataAllVal.push(dataVal);
  };
}

  const logs = await Promise.all(
    Range(0, numberOfUsers).map(async (userIndex) => 
      await runUser(task, url, dataAll[userIndex], userIndex, dataAllPush[userIndex], dataAllVal[userIndex])).toArray()
  )
  if (args.save) {
    const fileName = `${task.id}_${numberOfUsers}users.csv`;
    await fs.writeFile(fileName, JSON.stringify(logs, null, 2));
  }

  console.log('Shutting down the server...')
  await new Promise((resolve, reject) => {
    server.once('close', resolve)
    server.close(reject)
  })
}

main(args.task, args.numberOfUsers, args.dataDir, args.numClasses, args.push, args.val).catch(console.error)
